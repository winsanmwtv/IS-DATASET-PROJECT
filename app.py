import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# ----- MAPPINGS for Demo Inputs and Display -----
route_mapping = {
    'BTS สายสุขุมวิท': 1,
    'BTS สายสีลม': 2,
    'MRT สายสีน้ำเงิน': 3,
    'Airport Rail Link': 4,
    'MRT สายสีม่วง': 5,
    'BTS สายสีทอง': 6,
    'SRT สายสีแดงเหนือ': 7,
    'SRT สายสีแดงตะวันตก': 8,
    'MRT สายสีเหลือง': 9,
    'MRT สายสีชมพู': 10
}
route_mapping2 = {
    'ศรีรัช': 1,
    'บางพลี-สุขสวัสดิ์': 2,
    'ทางหลวงพิเศษหมายเลข 37': 3,
    'บูรพาวิถี': 4,
    'ฉลองรัช': 5,
    'เฉลิมมหานคร': 6,
    'อุดรรัถยา': 7,
    'S1': 8,
    'ศรีรัช-วงแหวนรอบนอก': 9
}

location_mapping = {
    'ภายในสถานี': 1,
    'ระหว่างสถานี': 2
}
weather_state = {
    'ปกติ': 0,
    'ฝนตก': 1
}


reason_mapping = {
    '01 ระบบขับเคลื่อน': 1,
    '02 ระบบเบรก': 2,
    '03 ระบบประตู': 3,
    '04 ระบบจ่ายไฟ': 4,
    '05 อุปกรณ์ในราง': 5,
    '06 จุดสับราง': 6,
    '07 ระบบนับเพลา': 7,
    '08 อาณัติสัญญาณ': 8,
    '09 วัสดุแปลกปลอม': 9,
    '10 อื่นๆ': 10,
    '11 ผู้โดยสาร': 11,
    '12 ระบบอัตโนมัติ': 12
}

cause_mapping = {
    'ขับรถเร็วเกินไป': 1,
    'เปลี่ยนช่องทางกระทันหัน': 2,
    'ขับรถกระชั้นชิดู': 3,
    'ยางแตก': 4,
    'ละสายตาจากการขับขี่': 5,
    'หลับใน': 6,
    'เบรกกะทันหัน': 7,
    'บรรทุกหนักเกินไป': 8,
    'ฝ่าฝืนสัญญาณ/ป้ายจราจร': 9,
    'เครื่องยนต์ขัดข้อง': 10,
    'เบรกขัดข้อง': 11,
    'ระบบไฟฟ้าขัดข้อง': 12,
    'ขับรถประมาท': 13
}

delay_mapping_manual = {
    'A': 'A ไม่มีผลกระทบ',
    'B': 'B น้อยกว่า 5 นาที',
    'C': 'C มากกว่า 5 นาที',
    'O': 'O null'
}

# ----- Load Data and Prepare Training Set -----
df = pd.read_csv('drt2565_06-2.csv')
df_railway_example = pd.read_csv('drt2565_06-2.csv')

df2 = pd.read_csv('accident.csv')
df2_accident_example= pd.read_csv('accident.csv')

df2['weather_state'] = df2['weather_state'].map(weather_state)
df2['cause'] = df2['cause'].map(cause_mapping)
df2['expw_step'] = df2['expw_step'].map(route_mapping2)

df2 = df2.dropna(subset=['dead_man'])  # Drop rows where 'dead_man' has NaN values
# Define the feature and target variables
X2 = df2[['expw_step', 'weather_state', 'cause']]  # Assuming your dataframe has these columns
y2 = df2['dead_man']

# Splitting data into training and testing sets
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X2_train = scaler.fit_transform(X2_train)
X2_test = scaler.transform(X2_test)

# Drop unused columns (including time-related ones)
df = df.drop(['ID', 'DATE', 'DETAIL', 'TRAIN_ID', 'ST1_ID', 'ST2_ID',
              'MONTH', 'YEAR', 'TIME', 'HOUR', 'MINS'], axis=1, errors='ignore')
#drop any null value
df2 = df2.dropna()
df2 = df2.drop(['accident_date','accident_time','injur_man','injur_femel','dead_femel'],axis=1,errors='ignore')

# Ensure numeric conversion for LINE_ID, PLACE_ID, and CAUSE_ID.
df['LINE_ID'] = pd.to_numeric(df['LINE_ID'], errors='coerce')
df['PLACE_ID'] = pd.to_numeric(df['PLACE_ID'], errors='coerce')
df['CAUSE_ID'] = pd.to_numeric(df['CAUSE_ID'], errors='coerce')

# Impute missing numeric values.
df.fillna(df.mean(numeric_only=True), inplace=True)

# Label-encode OPERATOR and DELAY_ID.
le_operator = LabelEncoder()
df['OPERATOR'] = le_operator.fit_transform(df['OPERATOR'])  # Keeps original OPERATOR values from CSV

le_delay = LabelEncoder()
df['DELAY_ID'] = le_delay.fit_transform(df['DELAY_ID'])
delay_classes = {i: cls for i, cls in enumerate(le_delay.classes_)}

# Feature set: LINE_ID, PLACE_ID, OPERATOR, CAUSE_ID.
X = df[['LINE_ID', 'PLACE_ID', 'OPERATOR', 'CAUSE_ID']]
y = df['DELAY_ID']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----- Improve Accuracy -----
model_rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=25,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42
)
model_rf.fit(X_train, y_train)

model_lr = LogisticRegression(
    max_iter=3000,
    C=0.3,
    solver="lbfgs"
)
model_lr.fit(X_train, y_train)

model_gb = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=10,
    random_state=42
)
model_gb.fit(X_train, y_train)

# ----- Define Streamlit Pages -----
def uc():
    st.title("หน้านี้ยังไม่พร้อมใช้งาน")

def main_page():
    st.title('Project')
    st.subheader('Machine Learning & Neural Network System')
    st.write('---')
    st.subheader('จัดทำโดย')
    st.write('- 6504062636195 ปรมะ ตันรัตนะ')
    st.write('- 6604062636453 พงศ์ศิริ เลิศพงษ์ไทย')
    st.write('---')
    st.subheader('แหล่งอ้างอิงข้อมูล')
    st.write('Machine Learning: https://data.go.th/dataset/stat_incident_metro')
    st.markdown('Neural Network: https://datagov.mot.go.th/en/dataset/exat-accident')

def desc_ml():
    st.title('Machine Learning')
    st.subheader('ข้อมูลการเกิดอุบัติเหตุทางรถไฟฟ้า')
    st.write('---')
    st.write('ข้อมูลการเกิดอุบัติเหตุทางรถไฟฟ้า ดึงมาจากเว็บของฐานข้อมูลรัฐ (data.go.th) ซึ่งมีการรวบรวมข้อมูลสถิติต่างๆ ไว้ เป็นไฟล์ .csv '
             'สามารถ download ออกมาใช้ได้ ซึ่งข้อมูลที่เราดึงมาจากลิงค์นี้ https://data.go.th/dataset/stat_incident_metro')
    st.write('---')
    st.subheader('เนื้อหาข้อมูล')
    st.write('ตามที่กล่าวไปข้างต้น ข้อมูลนี้จะรวบรวมสถิติการเกิดอุบัติเหตุทางรถไฟฟ้าใน กทม. โดยเก็บจากกรมการขนส่งทางราง ซึ่งข้อมูลมีความไม่เรียบร้อย '
             'มีค่า null รวมถึง ID เป็น null ด้วย และข้อมูลนี้จะเก็บรายละเอียด เช่น')
    st.write('- ประเภทอุบัติเหตุ (ประตูขัดข้อง, รางชำรุด, เหตุจากผู้โดยสาร, etc.)')
    st.write('- ระยะเวลาในการเกิดเหตุ เช่น เหตุเกิดขึ้นน้อยกว่า 5 นาที (จนแก้ไขได้สำเร็จ)')
    st.write('---')
    st.subheader('ฟีเจอร์ที่ดึงมาใช้')
    st.write('- รหัสสาย (สาย BTS, MRT สายสีน้ำเงิน, MRT สายสีม่วง, etc.)')
    st.write('- รหัสที่เกิดเหตุ (เกิดในสถานี นอกสถานี)')
    st.write('- ความล่าช้า (ไม่ระบุ, น้อยกว่า 5 นาที, มากกว่า 5 นาที)')
    st.write('- ผู้ให้บริการ')
    st.write('- ประเภทอุบัติเหตุ')
    st.write('---')
    st.subheader('ตัวอย่างข้อมูล')
    st.dataframe(df_railway_example.head())

def demo_ml():
    st.title('Demo (Machine Learning)')
    st.subheader('การทำนายความล่าช้าจากอุบัติเหตุทางรถไฟฟ้า')
    model_choice = st.selectbox('เลือกโมเดล', ['Random Forest', 'Logistic Regression'])

    # Input widgets for route, location, and reason.
    selected_route = st.selectbox('Route (เส้นทาง)', list(route_mapping.keys()))
    selected_location = st.radio('Location (สถานที่)', list(location_mapping.keys()))
    selected_reason = st.selectbox('Reason (สาเหตุ)', list(reason_mapping.keys()))

    if st.button('ทำนาย'):
        # Build input data using our mappings.
        input_data = pd.DataFrame([{
            'LINE_ID': route_mapping[selected_route],
            'PLACE_ID': location_mapping[selected_location],
            'OPERATOR': df[df['LINE_ID'] == route_mapping[selected_route]]['OPERATOR'].mode()[0],  # Use OPERATOR from CSV
            'CAUSE_ID': reason_mapping[selected_reason]
        }])

        if model_choice == 'Random Forest':
            prediction = model_rf.predict(input_data)
            y_pred = model_rf.predict(X_test)
            acc = accuracy_score(y_test, model_rf.predict(X_test)) * 100
        else:
            prediction = model_lr.predict(input_data)
            y_pred = model_lr.predict(X_test)
            acc = accuracy_score(y_test, model_lr.predict(X_test)) * 100

        pred_code = delay_classes[prediction[0]]
        pred_delay = delay_mapping_manual.get(pred_code, pred_code)

        st.write(f'ผลการทำนาย ({model_choice}): {pred_delay}')
        st.write(f'Accuracy: {acc:.2f}%')

        cm = confusion_matrix(y_test, y_pred)
        st.write('Confusion Matrix:')
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', ax=ax)
        st.pyplot(fig)

        if model_choice == 'Random Forest':
            st.write('Feature Importance:')
            feature_importance = pd.Series(model_rf.feature_importances_, index=X_train.columns)
            fig_chart = plt.figure()
            feature_importance.sort_values().plot(kind='barh')
            st.pyplot(fig_chart)
        else:
            st.write('Feature Importance ไม่พร้อมใช้งานสำหรับ Logistic Regression.')

        st.write('Prediction Result Chart:')
        prediction_counts = pd.Series(y_pred).value_counts().sort_index()
        fig_prediction = plt.figure()
        prediction_counts.plot(kind='bar')
        st.pyplot(fig_prediction)

def desc_nn():
    st.title('Neural Network')
    st.subheader('ข้อมูลการเกิดอุบัติเหตุทางพิเศษ (Neural Network)')
    st.write('---')
    st.write('ข้อมูลนี้ใช้ในการพัฒนาโมเดล Neural Network เช่น FNN หรือ CNN สำหรับการทำนายอุบัติเหตุจากทางพิเศษ')
    st.write('---')
    st.write(
        'ข้อมูลการเกิดอุบัติเหตุทางพิเศษ ดึงมาจากเว็บของกระทรวงคมนาคม การทางพิเศษแห่งประเทศไทย (datagov.mot.go.th) ซึ่งมีการรวบรวมข้อมูลสถิติต่างๆ ไว้ เป็นไฟล์ .csv '
        'สามารถ download ออกมาใช้ได้ ซึ่งข้อมูลที่เราดึงมาจากลิงค์นี้ที่เป็นในปี 2565 และข้อมูลนี้ได้มีการเพิ่มค่าว่างเพื่อทำให้เป็นข้อมูลที่ไม่สมบูรณ์ https://datagov.mot.go.th/en/dataset/exat-accident')
    st.write('---')
    st.subheader('ฟีเจอร์ที่ดึงมาใช้')
    st.write('- วันที่')
    st.write('- เวลา')
    st.write('- สถานที่เกิดเหตุ (ศรีรัช, บางพลี-สุขสวัสดิ, เฉลิมมหานคร, etc.)')
    st.write('- สถานะฤดู (ปกติ,ฝนตก)')
    st.write('- ประเภทอุบัติเหตุ')
    st.write('---')
    st.subheader('ตัวอย่างข้อมูล')
    st.dataframe(df2_accident_example.head())


def demo_nn():
    st.title('Demo (Neural Network)')
    st.subheader('การทำนายความล่าช้าจากอุบัติเหตุทางพิเศษ ด้วย Neural Network')

    # Select between FNN and CNN
    model_choice = st.selectbox('เลือกโมเดล Neural Network',
                                ['Feedforward Neural Network (FNN)', 'Convolutional Neural Network (CNN)'])

    # Input widgets for route, location, and reason
    selected_location_2 = st.selectbox('สถานที่เกิดเหตุ', list(route_mapping2.keys()))
    selected_weather_state = st.radio('สถานะฤดู (Weather)', list(weather_state.keys()))
    selected_cause = st.selectbox('ประเภทอุบัติเหตุ', list(cause_mapping.keys()))

    # Define model once outside the button (train the model outside)
    nn_model_fnn = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X2_train.shape[1],)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    nn_model_fnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history_fnn = nn_model_fnn.fit(X2_train, y2_train, epochs=10, batch_size=32,
                                   validation_data=(X2_test, y2_test))  # Train once

    # Train the CNN model outside if needed, or comment out if not needed
    nn_model_cnn = keras.Sequential([
        layers.Reshape((X2_train.shape[1], 1), input_shape=(X2_train.shape[1],)),
        layers.Conv1D(64, 2, activation='relu'),
        layers.MaxPooling1D(),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Or any other suitable output layer
    ])

    nn_model_cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history_cnn = nn_model_cnn.fit(X2_train, y2_train, epochs=10, batch_size=32,
                                   validation_data=(X2_test, y2_test))  # Train once

    if st.button('ทำนาย'):
        # Build input data using our mappings
        input_data2 = pd.DataFrame([{
            'expw_step': route_mapping2[selected_location_2],
            'weather_state': weather_state[selected_weather_state],
            'cause': cause_mapping[selected_cause]
        }])
        input_data_nn = scaler.transform(input_data2)

        if model_choice == 'Feedforward Neural Network (FNN)':
            prediction_nn = nn_model_fnn.predict(input_data_nn)
            prediction_result_nn = 'High' if prediction_nn[0] > 0.5 else 'Low'  # Thresholded prediction
            st.write(f'ผลการทำนาย (FNN): {prediction_result_nn}')

        elif model_choice == 'Convolutional Neural Network (CNN)':
            prediction_nn = nn_model_cnn.predict(input_data_nn)
            prediction_result_nn = 'High' if prediction_nn[0] > 0.5 else 'Low'  # Thresholded prediction
            st.write(f'ผลการทำนาย (CNN): {prediction_result_nn}')

        st.write('Prediction complete with Neural Network!')

        # Plot training and validation accuracy
        if model_choice == 'Feedforward Neural Network (FNN)':
            st.subheader('FNN Accuracy Over Epochs')
            plt.figure(figsize=(8, 6))
            plt.plot(history_fnn.history['accuracy'], label='Training Accuracy')
            plt.plot(history_fnn.history['val_accuracy'], label='Validation Accuracy')
            plt.title('FNN Model Accuracy Over Epochs')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            st.pyplot()

        elif model_choice == 'Convolutional Neural Network (CNN)':
            st.subheader('CNN Accuracy Over Epochs')
            plt.figure(figsize=(8, 6))
            plt.plot(history_cnn.history['accuracy'], label='Training Accuracy')
            plt.plot(history_cnn.history['val_accuracy'], label='Validation Accuracy')
            plt.title('CNN Model Accuracy Over Epochs')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            st.pyplot()

        # Optionally, plot the confusion matrix
        y_pred_fnn = nn_model_fnn.predict(X2_test)
        y_pred_fnn = (y_pred_fnn > 0.5).astype(int)
        cm_fnn = confusion_matrix(y2_test, y_pred_fnn)

        st.subheader('Confusion Matrix for FNN')
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_fnn, annot=True, fmt='d', cmap='Blues', xticklabels=['B', 'A'], yticklabels=['B', 'A'])
        plt.title('FNN Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        st.pyplot()




# ----- Navigation -----
pages = {
    'หน้าหลัก': main_page,
    'คำอธิบาย Machine Learning': desc_ml,
    'คำอธิบาย Neural Network': desc_nn,
    'Demo: Machine Learning': demo_ml,
    'Demo: Neural Network': demo_nn
}

selected_page = st.sidebar.selectbox('เลือกหน้า', list(pages.keys()))
pages[selected_page]()
