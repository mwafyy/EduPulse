import sqlite3
import os
import json
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "edupulse.db")
DATA_DIR = os.path.join(BASE_DIR, "data")

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    # Students table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_code TEXT UNIQUE,
        name TEXT,
        course TEXT,
        school TEXT,
        sex TEXT,
        age INTEGER,
        studytime INTEGER,
        absences INTEGER,
        failures INTEGER,
        schoolsup TEXT,
        famsup TEXT,
        paid TEXT,
        activities TEXT,
        higher TEXT,
        internet TEXT,
        health INTEGER,
        goout INTEGER,
        freetime INTEGER,
        traveltime INTEGER,
        risk_level TEXT,
        risk_probability REAL,
        raw_features TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # Interventions table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS interventions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER,
        title TEXT,
        category TEXT,
        status TEXT DEFAULT 'Pending',
        notes TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(student_id) REFERENCES students(id)
    )
    ''')

    conn.commit()
    conn.close()

def seed_db_if_empty(limit=150):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM students")
    count = cursor.fetchone()[0]

    if count == 0:
        print("Seeding database with sample students from dataset...")
        from ml.predict import predict_single

        # Load sample students
        mat_path = os.path.join(DATA_DIR, "student-mat.csv")
        por_path = os.path.join(DATA_DIR, "student-por.csv")

        df_mat = pd.read_csv(mat_path, sep=';')
        df_mat['course'] = 'Math'
        df_por = pd.read_csv(por_path, sep=';')
        df_por['course'] = 'Portuguese'

        df = pd.concat([df_mat, df_por], ignore_index=True)
        sample = df.sample(n=min(limit, len(df)), random_state=42)

        arabic_first_names = ["أحمد", "محمد", "عمر", "يوسف", "علي", "سارة", "نور", "مريم", "فاطمة", "شهد", "كريم", "زياد", "سلمى", "آية", "حمزة", "ليلى"]
        arabic_last_names = ["محمود", "حسن", "إبراهيم", "الشريف", "خالد", "منصور", "عبد الرحمن", "سالم", "طارق", "فهمي", "صلاح", "النجار"]

        leakage_cols = ['G1', 'G2', 'G3']
        feature_cols = [c for c in sample.columns if c not in leakage_cols]

        idx = 1001
        for _, row in sample.iterrows():
            feat_dict = {col: row[col] for col in feature_cols}
            res = predict_single(feat_dict)

            f_name = arabic_first_names[idx % len(arabic_first_names)]
            l_name = arabic_last_names[(idx * 3) % len(arabic_last_names)]
            full_name = f"{f_name} {l_name}"
            code = f"STU-{idx}"

            cursor.execute('''
            INSERT INTO students (
                student_code, name, course, school, sex, age,
                studytime, absences, failures, schoolsup, famsup,
                paid, activities, higher, internet, health, goout, freetime, traveltime,
                risk_level, risk_probability, raw_features
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                code, full_name, row['course'], row['school'], row['sex'], int(row['age']),
                int(row['studytime']), int(row['absences']), int(row['failures']),
                row['schoolsup'], row['famsup'], row['paid'], row['activities'],
                row['higher'], row['internet'], int(row['health']), int(row['goout']),
                int(row['freetime']), int(row['traveltime']),
                res['risk_level'], res['risk_probability'], json.dumps(feat_dict, ensure_ascii=False)
            ))

            student_id = cursor.lastrowid
            # Seed 1 initial recommended intervention for high/medium risk
            if res['risk_level'] in ['High Risk', 'Medium Risk'] and res['recommendations']:
                rec = res['recommendations'][0]
                cursor.execute('''
                INSERT INTO interventions (student_id, title, category, status, notes)
                VALUES (?, ?, ?, 'Pending', ?)
                ''', (student_id, rec['action'], rec['category'], "إجراء مقترح آلياً بواسطة نظام الإنذار المبكر"))

            idx += 1

        conn.commit()
        print(f"Database successfully seeded with {len(sample)} students.")

    conn.close()

if __name__ == "__main__":
    init_db()
    seed_db_if_empty()
