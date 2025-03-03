import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df_railway = pd.read_csv('drt2565_06-2.csv')

# ตรวจสอบค่าว่าง
print(df_railway.isnull().sum())
df_railway.dropna(inplace=True)
#
# # แปลงประเภทข้อมูล (ถ้าจำเป็น)
# df_railway['DATE'] = pd.to_datetime(df_railway['DATE']) # แปลงคอลัมน์ DATE เป็น datetime
# df_railway['TIME'] = pd.to_numeric(df_railway['TIME'], errors='coerce') # แปลงคอลัมน์ TIME เป็น numeric และจัดการค่าที่ไม่ถูกต้อง
#
# # จัดการข้อมูลเชิงหมวดหมู่ (Encoding)
# le_line = LabelEncoder()
# df_railway['LINE_ID'] = le_line.fit_transform(df_railway['LINE_ID'])
#
# le_delay = LabelEncoder()
# df_railway['DELAY_ID'] = le_delay.fit_transform(df_railway['DELAY_ID'])
#
# le_operator = LabelEncoder()
# df_railway['OPERATOR'] = le_operator.fit_transform(df_railway['OPERATOR'])
#
# le_cause = LabelEncoder()
# df_railway['CAUSE_ID'] = le_cause.fit_transform(df_railway['CAUSE_ID'])
#
# # สร้างคอลัมน์ใหม่จาก DATE (เช่น ปี, เดือน, วัน)
# df_railway['YEAR'] = df_railway['DATE'].dt.year
# df_railway['MONTH_NUM'] = df_railway['DATE'].dt.month
# df_railway['DAY'] = df_railway['DATE'].dt.day
#
# # สร้างคอลัมน์ใหม่จาก TIME (เช่น ช่วงเวลา)
# df_railway['TIME_BIN'] = pd.cut(df_railway['TIME'], bins=[0, 600, 1200, 1800, 2400], labels=['Morning', 'Noon', 'Afternoon', 'Night'])
# le_time_bin = LabelEncoder()
# df_railway['TIME_BIN'] = le_time_bin.fit_transform(df_railway['TIME_BIN'])
#
# X_railway = df_railway.drop(['ID', 'DATE', 'DETAIL'], axis=1) # สร้าง X โดยไม่รวมคอลัมน์ที่ไม่จำเป็น
# y_railway = df_railway['DELAY_ID'] # สร้าง y จากคอลัมน์ DELAY_ID
# X_train_railway, X_test_railway, y_train_railway, y_test_railway = train_test_split(X_railway, y_railway, test_size=0.2, random_state=42)
#
# X_train_railway.to_csv('X_train_railway.csv', index=False)
# X_test_railway.to_csv('X_test_railway.csv', index=False)
# y_train_railway.to_csv('y_train_railway.csv', index=False)
# y_test_railway.to_csv('y_test_railway.csv', index=False)