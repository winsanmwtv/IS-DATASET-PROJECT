import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import re

# โหลดข้อมูล CSV
df_railway = pd.read_csv('drt2565_06-2.csv')

# แปลง MONTH เป็นตัวเลข
month_mapping = {
    'มกราคม': 1, 'กุมภาพันธ์': 2, 'มีนาคม': 3, 'เมษายน': 4, 'พฤษภาคม': 5, 'มิถุนายน': 6,
    'กรกฎาคม': 7, 'สิงหาคม': 8, 'กันยายน': 9, 'ตุลาคม': 10, 'พฤศจิกายน': 11, 'ธันวาคม': 12
}
df_railway['MONTH'] = df_railway['MONTH'].map(month_mapping)
df_railway['MONTH'].fillna(df_railway['MONTH'].mean(), inplace=True)

# แปลง TIME เป็น HOUR และ MINS
def parse_time(time_str):
    if pd.isnull(time_str):
        return None, None
    if isinstance(time_str, (int, float)):
        hour = int(time_str)
        return hour, 0
    match = re.match(r'(\d+)(?:\.(\d+))?', str(time_str))
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2)) if match.group(2) else 0
        return hour, minute
    return None, None

df_railway[['HOUR', 'MINS']] = df_railway['TIME'].apply(lambda x: pd.Series(parse_time(x)))

# จัดการค่าว่างใน TIME
df_railway.dropna(subset=['HOUR', 'MINS'], inplace=True)
df_railway.drop('TIME', axis=1, inplace=True)

# จัดการค่าว่างใน HOUR และ MINS (หลังจากลบแถว)
df_railway['HOUR'].fillna(df_railway['HOUR'].mean(), inplace=True)
df_railway['MINS'].fillna(df_railway['MINS'].mean(), inplace=True)

# สร้าง YEAR
df_railway['YEAR'] = pd.to_datetime(df_railway['DATE']).dt.year

# เตรียมข้อมูลสำหรับ Machine Learning
le_line = LabelEncoder()
df_railway['LINE_ID'] = le_line.fit_transform(df_railway['LINE_ID'])
le_delay = LabelEncoder()
df_railway['DELAY_ID'] = le_delay.fit_transform(df_railway['DELAY_ID'])
le_operator = LabelEncoder()
df_railway['OPERATOR'] = le_operator.fit_transform(df_railway['OPERATOR'])
le_cause = LabelEncoder()
df_railway['CAUSE_ID'] = le_cause.fit_transform(df_railway['CAUSE_ID'])

# ตัดคอลัมน์ที่ไม่ต้องการ (ไม่รวม DELAY_ID, YEAR, MONTH, HOUR, MINS)
X_railway = df_railway.drop(['ID', 'DATE', 'DETAIL', 'TRAIN_ID', 'ST1_ID', 'ST2_ID'], axis=1)
y_railway = df_railway['DELAY_ID']
X_train_railway, X_test_railway, y_train_railway, y_test_railway = train_test_split(X_railway, y_railway, test_size=0.2, random_state=42)

# สร้างและฝึกโมเดล Machine Learning (Random Forest)
model_rf = RandomForestClassifier()
model_rf.fit(X_train_railway, y_train_railway)

# หน้า 1: อธิบาย Machine Learning (อุบัติเหตุทางรถไฟฟ้า)
def page1():
    st.title('Machine Learning: การทำนายอุบัติเหตุทางรถไฟฟ้า')
    st.write('อธิบายแนวทางการพัฒนา, ทฤษฎีของอัลกอริทึม, และขั้นตอนการพัฒนาโมเดล')
    st.write('ข้อมูลที่ใช้: ข้อมูลอุบัติเหตุทางรถไฟฟ้า (CSV)')
    st.dataframe(df_railway.head())
    st.write('โมเดลที่ใช้: Random Forest')
    st.write('อัลกอริทึม Random Forest เป็นอัลกอริทึมที่ใช้การตัดสินใจแบบต้นไม้หลายต้นมาช่วยในการตัดสินใจ')
    st.write('ขั้นตอนการพัฒนา: เตรียมข้อมูล, สร้างและฝึกโมเดล, ประเมินโมเดล')

# หน้า 2: อธิบาย Neural Network (ตรวจจับหลุมถนน)
def page2():
    st.title('Neural Network: การตรวจจับหลุมถนนจากรูปภาพ (จำลอง)')
    st.write('อธิบายแนวทางการพัฒนา, ทฤษฎีของอัลกอริทึม, และขั้นตอนการพัฒนาโมเดล')
    st.write('เนื่องจากไม่มีโมเดล Neural Network ที่ฝึกไว้ จะแสดงผลการทำงานจำลอง')
    st.write('ขั้นตอนการพัฒนา: รวบรวมข้อมูลรูปภาพ, เตรียมข้อมูล, สร้างและฝึกโมเดล, ประเมินโมเดล')
    st.write('โมเดล Neural Network ที่ใช้: Convolutional Neural Network (CNN)')
    st.write('CNN เป็นโมเดลที่เหมาะสมกับการประมวลผลรูปภาพ')

# หน้า 3: Demo Machine Learning (อุบัติเหตุทางรถไฟฟ้า)
def page3():
    st.title('Demo: การทำนายอุบัติเหตุทางรถไฟฟ้า')
    line_id = st.selectbox('เส้นทาง', X_train_railway['LINE_ID'].unique())
    place_id = st.radio('สถานที่', [1, 2])
    operator = st.selectbox('ผู้ให้บริการ', X_train_railway['OPERATOR'].unique())
    cause_id = st.selectbox('สาเหตุ', X_train_railway['CAUSE_ID'].unique())

    if st.button('ทำนาย'):
        input_data = pd.DataFrame([[line_id, place_id, operator, cause_id]],
                                 columns=['LINE_ID', 'PLACE_ID', 'OPERATOR', 'CAUSE_ID'])
        input_data = input_data[X_train_railway.columns]
        prediction = model_rf.predict(input_data)
        st.write(f'ผลการทำนาย: {prediction[0]}')

        # Confusion Matrix
        y_pred = model_rf.predict(X_test_railway)
        cm = confusion_matrix(y_test_railway, y_pred)
        st.write('Confusion Matrix:')
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', ax=ax)
        st.pyplot(fig)

        # Chart
        st.write('Feature Importance:')
        feature_importance = pd.Series(model_rf.feature_importances_, index=X_train_railway.columns)
        fig_chart = plt.figure()
        feature_importance.sort_values().plot(kind='barh')
        st.pyplot(fig_chart)

# หน้า 4: Demo Neural Network (ตรวจจับหลุมถนน)
def page4():
    st.title('Demo: การตรวจจับหลุมถนนจากรูปภาพ (จำลอง)')
    st.write('อธิบายแนวทางการพัฒนา, ทฤษฎีของอัลกอริทึม, และขั้นตอนการพัฒนาโมเดล')
    st.write('เนื่องจากไม่มีโมเดล Neural Network ที่ฝึกไว้ จะแสดงผลการทำงานจำลอง')
    st.write('ขั้นตอนการพัฒนา: รวบรวมข้อมูลรูปภาพ, เตรียมข้อมูล, สร้างและฝึกโมเดล, ประเมินโมเดล')
    st.write('โมเดล Neural Network ที่ใช้: Convolutional Neural Network (CNN)')
    st.write('CNN เป็นโมเดลที่เหมาะสมกับการประมวลผลรูปภาพ')

# สร้าง Navigation
pages = {
    'Machine Learning': page1,
    'Neural Network': page2,
    'Demo Machine Learning': page3,
    'Demo Neural Network': page4
}

selected_page = st.sidebar.selectbox('เลือกหน้า', list(pages.keys()))
pages[selected_page]()