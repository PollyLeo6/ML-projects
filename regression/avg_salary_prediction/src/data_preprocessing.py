import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_preprocess_data(data_path):
    df = pd.read_csv(data_path)
    
    # Извлечение зарплат из JSON
    def extract_salary(salary_str):
        if pd.isna(salary_str) or salary_str == 'null':
            return None
        try:
            salary_dict = json.loads(salary_str.replace("'", '"').replace('null', 'None'))
            salary_from = salary_dict.get('salary_from')
            salary_to = salary_dict.get('salary_to')
            
            if salary_from and salary_to:
                return (salary_from + salary_to) / 2
            elif salary_from:
                return salary_from
            elif salary_to:
                return salary_to
            return None
        except:
            return None
    
    df['avg_salary'] = df['salary'].apply(extract_salary)
    df = df.dropna(subset=['avg_salary'])
    
    # Кодирование опыта
    experience_map = {'0': 0, '1-3': 2, '3-6': 4.5, '6+': 8}
    df['experience_years'] = df['experience'].map(experience_map).fillna(0)
    
    # Кодирование категориальных переменных
    le_schedule = LabelEncoder()
    le_employment = LabelEncoder()
    
    df['schedule_encoded'] = le_schedule.fit_transform(df['schedule'].fillna('unknown'))
    df['employment_encoded'] = le_employment.fit_transform(df['employment'].fillna('unknown'))
    
    # Количество навыков
    def count_skills(skills_str):
        if pd.isna(skills_str):
            return 0
        try:
            skills = eval(skills_str)
            return len(skills) if isinstance(skills, list) else 0
        except:
            return 0
    
    df['skills_count'] = df['key_skills'].apply(count_skills)
    
    # Подготовка признаков
    features = ['experience_years', 'schedule_encoded', 'employment_encoded', 'skills_count']
    X = df[features]
    y = df['avg_salary']
    
    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Масштабирование
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler