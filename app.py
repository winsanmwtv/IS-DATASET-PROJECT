import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

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

location_mapping = {
    'ภายในสถานี': 1,
    'ระหว่างสถานี': 2
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

delay_mapping_manual = {
    'A': 'A ไม่มีผลกระทบ',
    'B': 'B น้อยกว่า 5 นาที',
    'C': 'C มากกว่า 5 นาที',
    'O': 'O null'
}

# ----- Load Data and Prepare Training Set -----
df = pd.read_csv('drt2565_06-2.csv')

# Drop unused columns (including time-related ones)
df = df.drop(['ID', 'DATE', 'DETAIL', 'TRAIN_ID', 'ST1_ID', 'ST2_ID',
              'MONTH', 'YEAR', 'TIME', 'HOUR', 'MINS'], axis=1, errors='ignore')

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

# Train models.
model_rf = RandomForestClassifier(random_state=42)
model_rf.fit(X_train, y_train)

model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train, y_train)

# ----- Define Streamlit Pages -----
def main_page():
    st.title('Project: Machine Learning & Neural Network System')
    st.write('จัดทำโดย ... ปรมะ')
    st.write('6504062636195 ปรมะ ตันรัตนะ')
    st.write('6604062636453 พงศ์ศิริ เลิศพงษ์ไทย')
    st.write('---')
    st.write('ในส่วนของ Demo จะใช้โมเดลสองแบบ (Random Forest และ Logistic Regression) '
             'เพื่อทำนายผลกระทบของความล่าช้า (Delay) โดยใช้ข้อมูล: '
             'เส้นทาง (Route), สถานที่ (Location), และ สาเหตุ (Reason).')
    st.write('Operator ยังคงใช้ค่าตามข้อมูลที่มีใน CSV และไม่ได้ให้ผู้ใช้เลือกเอง.')

def demo_ml():
    st.title('Demo: การทำนายอุบัติเหตุทางรถไฟฟ้า')
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

# ----- Navigation -----
pages = {
    'หน้าหลัก': main_page,
    'Demo Machine Learning': demo_ml
}

selected_page = st.sidebar.selectbox('เลือกหน้า', list(pages.keys()))
pages[selected_page]()
