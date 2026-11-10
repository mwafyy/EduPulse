
import os
import sys
import json
import hmac
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import (
    VerifyMismatchError,
    VerificationError,
    InvalidHashError
)

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.database import get_connection
from ml.predict import predict_single, generate_recommendations


# ============================================================
# SECURITY CONFIGURATION
# ============================================================

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 8

password_hasher = PasswordHasher()
bearer_scheme = HTTPBearer(auto_error=False)

app = FastAPI(
    title="EduPulse AI API",
    description="Multi-Tenant Early Warning & Student Performance Intervention Platform",
    version="3.1.0"
)


# ============================================================
# TEACHER COURSE ASSIGNMENTS
# ============================================================

# Teacher ID 2: mwafyy -> Math
# Teacher ID 3: mwafy  -> Portuguese
#
# Admin can access all courses.

TEACHER_COURSES = {
    2: "Math",
    3: "Portuguese"
}


def get_teacher_course(teacher_id: int):
    """Return the course assigned to a teacher."""

    return TEACHER_COURSES.get(teacher_id)


def validate_teacher_course(teacher_id: int, course: str):
    """Ensure a teacher is assigned to the requested course."""

    assigned_course = get_teacher_course(teacher_id)

    if assigned_course is None:
        raise HTTPException(
            status_code=403,
            detail="No course assigned to this teacher"
        )

    if course != assigned_course:
        raise HTTPException(
            status_code=403,
            detail=f"This teacher can only manage {assigned_course}"
        )

    return assigned_course


# ============================================================
# CORS
# ============================================================

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://127.0.0.1:8000,http://localhost:8000"
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


# ============================================================
# REQUEST MODELS
# ============================================================

class LoginRequest(BaseModel):
    username: str
    password: str


class StudentInput(BaseModel):
    name: str
    student_code: Optional[str] = None
    teacher_id: Optional[int] = 2

    course: str = Field(default="Math")
    school: str = Field(default="GP")
    sex: str = Field(default="M")

    age: int = Field(default=16, ge=12, le=30)
    studytime: int = Field(default=2, ge=1, le=4)
    absences: int = Field(default=4, ge=0, le=100)
    failures: int = Field(default=0, ge=0, le=5)

    schoolsup: str = Field(default="no")
    famsup: str = Field(default="yes")
    paid: str = Field(default="no")
    activities: str = Field(default="no")
    higher: str = Field(default="yes")
    internet: str = Field(default="yes")

    health: int = Field(default=4, ge=1, le=5)
    goout: int = Field(default=3, ge=1, le=5)
    freetime: int = Field(default=3, ge=1, le=5)
    traveltime: int = Field(default=1, ge=1, le=4)


class InterventionCreate(BaseModel):
    student_id: int
    title: str
    category: str
    notes: Optional[str] = "تم تسجيل التدخل من لوحة المتابعة"


# ============================================================
# AUTHENTICATION
# ============================================================

def create_access_token(user_id: int, role: str) -> str:

    if not JWT_SECRET_KEY:
        raise HTTPException(
            status_code=500,
            detail="Server security is not configured. Set JWT_SECRET_KEY."
        )

    now = datetime.now(timezone.utc)

    payload = {
        "sub": str(user_id),
        "role": role,
        "iat": now,
        "exp": now + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    }

    return jwt.encode(
        payload,
        JWT_SECRET_KEY,
        algorithm=JWT_ALGORITHM
    )


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        bearer_scheme
    )
):

    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"}
        )

    if not JWT_SECRET_KEY:
        raise HTTPException(
            status_code=500,
            detail="Server security is not configured"
        )

    try:
        payload = jwt.decode(
            credentials.credentials,
            JWT_SECRET_KEY,
            algorithms=[JWT_ALGORITHM]
        )

        user_id = int(payload.get("sub"))
        role = payload.get("role")

        if role not in ("admin", "teacher"):
            raise ValueError("Invalid role")

    except (jwt.InvalidTokenError, TypeError, ValueError):
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"}
        )

    conn = get_connection()

    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id, username, name, school_name, role
            FROM teachers
            WHERE id = ?
            """,
            (user_id,)
        )

        row = cursor.fetchone()

    finally:
        conn.close()

    if not row or row["role"] != role:
        raise HTTPException(
            status_code=401,
            detail="User account is not valid"
        )

    return dict(row)


def require_admin(current_user: dict = Depends(get_current_user)):

    if current_user["role"] != "admin":
        raise HTTPException(
            status_code=403,
            detail="Admin access required"
        )

    return current_user


# ============================================================
# STUDENT ACCESS CONTROL
# ============================================================

def ensure_student_access(cursor, student_id: int, current_user: dict):
    """
    Admin can access all students.

    Teachers can access only students that:
    1. Belong to their teacher ID.
    2. Match their assigned course.
    """

    cursor.execute(
        "SELECT * FROM students WHERE id = ?",
        (student_id,)
    )

    student = cursor.fetchone()

    if not student:
        raise HTTPException(
            status_code=404,
            detail="Student not found"
        )

    if current_user["role"] == "admin":
        return student

    assigned_course = get_teacher_course(current_user["id"])

    if assigned_course is None:
        raise HTTPException(
            status_code=403,
            detail="No course assigned to this teacher"
        )

    if (
        student["teacher_id"] != current_user["id"]
        or student["course"] != assigned_course
    ):
        raise HTTPException(
            status_code=404,
            detail="Student not found"
        )

    return student


# ============================================================
# PASSWORD VERIFICATION
# ============================================================

def verify_and_upgrade_password(
    stored_password: str,
    supplied_password: str
) -> bool:

    if stored_password.startswith("$argon2"):

        try:
            return password_hasher.verify(
                stored_password,
                supplied_password
            )

        except (
            VerifyMismatchError,
            VerificationError,
            InvalidHashError
        ):
            return False

    # Legacy plaintext password transition.

    if not hmac.compare_digest(
        str(stored_password),
        str(supplied_password)
    ):
        return False

    return True


# ============================================================
# ML FEATURE HELPER
# ============================================================

def fill_features(data: dict):

    data.setdefault("address", "U")
    data.setdefault("famsize", "GT3")
    data.setdefault("Pstatus", "T")

    data.setdefault("Medu", 2)
    data.setdefault("Fedu", 2)

    data.setdefault("Mjob", "other")
    data.setdefault("Fjob", "other")

    data.setdefault("reason", "course")
    data.setdefault("guardian", "mother")

    data.setdefault("nursery", "yes")
    data.setdefault("romantic", "no")

    data.setdefault("famrel", 4)
    data.setdefault("Dalc", 1)
    data.setdefault("Walc", 1)

    return data


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/api/health")
def health_check():

    return {
        "status": "ok",
        "app": "EduPulse AI Early Warning System",
        "version": "3.1.0"
    }


# ============================================================
# LOGIN
# ============================================================

@app.post("/api/auth/login")
def login(req: LoginRequest):

    conn = get_connection()

    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id, username, password, name, school_name, role
            FROM teachers
            WHERE username = ?
            """,
            (req.username,)
        )

        row = cursor.fetchone()

        if not row or not verify_and_upgrade_password(
            row["password"],
            req.password
        ):
            raise HTTPException(
                status_code=401,
                detail="اسم المستخدم أو كلمة المرور غير صحيحة"
            )

        # Upgrade old plaintext passwords to Argon2.

        if not row["password"].startswith("$argon2"):

            new_hash = password_hasher.hash(req.password)

            cursor.execute(
                "UPDATE teachers SET password = ? WHERE id = ?",
                (new_hash, row["id"])
            )

            conn.commit()

        token = create_access_token(
            row["id"],
            row["role"]
        )

        return {
            "status": "success",
            "access_token": token,
            "token_type": "bearer",
            "expires_in": ACCESS_TOKEN_EXPIRE_HOURS * 3600,
            "teacher": {
                "id": row["id"],
                "username": row["username"],
                "name": row["name"],
                "school_name": row["school_name"],
                "role": row["role"]
            }
        }

    finally:
        conn.close()


# ============================================================
# TEACHERS
# ============================================================

@app.get("/api/teachers")
def list_teachers(
    current_user: dict = Depends(require_admin)
):

    conn = get_connection()

    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id, username, name, school_name, role
            FROM teachers
            ORDER BY id
            """
        )

        teachers = [
            dict(row)
            for row in cursor.fetchall()
        ]

        return {"teachers": teachers}

    finally:
        conn.close()


# ============================================================
# MODEL METRICS
# ============================================================

@app.get("/api/metrics")
def get_model_metrics(
    current_user: dict = Depends(get_current_user)
):

    metrics_path = os.path.join(
        BASE_DIR,
        "ml",
        "saved_models",
        "metrics_and_metadata.json"
    )

    if not os.path.exists(metrics_path):
        raise HTTPException(
            status_code=404,
            detail="Model metrics not found."
        )

    with open(metrics_path, "r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# DASHBOARD SUMMARY
# ============================================================

@app.get("/api/dashboard/summary")
def get_dashboard_summary(
    teacher_id: Optional[int] = None,
    current_user: dict = Depends(get_current_user)
):
    conn = get_connection()

    try:
        cursor = conn.cursor()

        filters = []
        params = []

        if current_user["role"] == "teacher":
            assigned_course = get_teacher_course(current_user["id"])

            if assigned_course is None:
                raise HTTPException(
                    status_code=403,
                    detail="No course assigned to this teacher"
                )

            filters.extend(["teacher_id = ?", "course = ?"])
            params.extend([current_user["id"], assigned_course])

        elif teacher_id is not None and teacher_id > 1:
            filters.append("teacher_id = ?")
            params.append(teacher_id)

        where_clause = (
            " WHERE " + " AND ".join(filters)
            if filters else ""
        )

        cursor.execute(
            f"SELECT COUNT(*) FROM students{where_clause}",
            params
        )
        total_students = cursor.fetchone()[0]

        risk_counts = {}

        for risk in ("High Risk", "Medium Risk", "Low Risk"):
            risk_filters = filters + ["risk_level = ?"]
            risk_params = params + [risk]

            cursor.execute(
                "SELECT COUNT(*) FROM students WHERE "
                + " AND ".join(risk_filters),
                risk_params
            )

            risk_counts[risk] = cursor.fetchone()[0]

        cursor.execute(
            f"""
            SELECT AVG(absences), AVG(studytime), AVG(failures)
            FROM students
            {where_clause}
            """,
            params
        )

        avg_row = cursor.fetchone()

        at_risk_total = (
            risk_counts["High Risk"] + risk_counts["Medium Risk"]
        )

        return {
            "total_students": total_students,
            "high_risk_count": risk_counts["High Risk"],
            "medium_risk_count": risk_counts["Medium Risk"],
            "low_risk_count": risk_counts["Low Risk"],
            "at_risk_total": at_risk_total,
            "risk_percentage": round(
                (at_risk_total / max(1, total_students)) * 100, 1
            ),
            "avg_absences": round(avg_row[0] or 0, 1),
            "avg_studytime": round(avg_row[1] or 0, 1),
            "avg_failures": round(avg_row[2] or 0, 2)
        }

    finally:
        conn.close()


# STUDENTS LIST
# ============================================================

@app.get("/api/students")
def get_students(
    teacher_id: Optional[int] = None,
    risk_level: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    current_user: dict = Depends(get_current_user)
):

    conn = get_connection()

    try:
        cursor = conn.cursor()

        query = "SELECT * FROM students WHERE 1=1"
        count_query = "SELECT COUNT(*) FROM students WHERE 1=1"

        params = []
        count_params = []

        if current_user["role"] == "teacher":

            assigned_course = get_teacher_course(
                current_user["id"]
            )

            if assigned_course is None:
                raise HTTPException(
                    status_code=403,
                    detail="No course assigned to this teacher"
                )

            query += " AND teacher_id = ? AND course = ?"
            count_query += " AND teacher_id = ? AND course = ?"

            params.extend([
                current_user["id"],
                assigned_course
            ])

            count_params.extend([
                current_user["id"],
                assigned_course
            ])

        elif teacher_id is not None and teacher_id > 1:

            query += " AND teacher_id = ?"
            count_query += " AND teacher_id = ?"

            params.append(teacher_id)
            count_params.append(teacher_id)

        if risk_level and risk_level != "All":

            query += " AND risk_level = ?"
            count_query += " AND risk_level = ?"

            params.append(risk_level)
            count_params.append(risk_level)

        if search:

            query += " AND (name LIKE ? OR student_code LIKE ?)"
            count_query += " AND (name LIKE ? OR student_code LIKE ?)"

            wildcard = f"%{search}%"

            params.extend([wildcard, wildcard])
            count_params.extend([wildcard, wildcard])

        cursor.execute(count_query, count_params)
        total_count = cursor.fetchone()[0]

        query += " ORDER BY id DESC LIMIT ? OFFSET ?"

        params.extend([limit, offset])

        cursor.execute(query, params)

        students = [
            dict(row)
            for row in cursor.fetchall()
        ]

        return {
            "students": students,
            "count": total_count
        }

    finally:
        conn.close()


# ============================================================
# STUDENT DETAILS
# ============================================================

@app.get("/api/students/{student_id}")
def get_student_detail(
    student_id: int,
    current_user: dict = Depends(get_current_user)
):

    conn = get_connection()

    try:
        cursor = conn.cursor()

        row = ensure_student_access(
            cursor,
            student_id,
            current_user
        )

        student = dict(row)

        cursor.execute(
            """
            SELECT *
            FROM interventions
            WHERE student_id = ?
            ORDER BY id DESC
            """,
            (student_id,)
        )

        student["interventions"] = [
            dict(item)
            for item in cursor.fetchall()
        ]

        raw_features = json.loads(
            student.get("raw_features") or "{}"
        )

        student["recommended_actions"] = generate_recommendations(
            raw_features,
            student["risk_level"],
            student["risk_probability"] / 100.0
        )

        return student

    finally:
        conn.close()


# ============================================================
# PREDICT
# ============================================================

@app.post("/api/predict")
def predict_risk(
    data: StudentInput,
    current_user: dict = Depends(get_current_user)
):

    if current_user["role"] == "teacher":

        validate_teacher_course(
            current_user["id"],
            data.course
        )

    feature_dict = fill_features(data.dict())

    return predict_single(feature_dict)


# ============================================================
# CREATE STUDENT
# ============================================================

@app.post("/api/students")
def create_student(
    data: StudentInput,
    current_user: dict = Depends(get_current_user)
):

    conn = get_connection()

    try:
        cursor = conn.cursor()

        if current_user["role"] == "teacher":

            assigned_teacher_id = current_user["id"]

            validate_teacher_course(
                assigned_teacher_id,
                data.course
            )

        else:

            assigned_teacher_id = data.teacher_id or 2

            cursor.execute(
                """
                SELECT id
                FROM teachers
                WHERE id = ? AND role = 'teacher'
                """,
                (assigned_teacher_id,)
            )

            if not cursor.fetchone():
                raise HTTPException(
                    status_code=400,
                    detail="Selected teacher does not exist"
                )

            validate_teacher_course(
                assigned_teacher_id,
                data.course
            )

        if not data.student_code:

            cursor.execute("SELECT MAX(id) FROM students")

            max_id = cursor.fetchone()[0] or 1000

            data.student_code = f"STU-{max_id + 1}"

        cursor.execute(
            "SELECT id FROM students WHERE student_code = ?",
            (data.student_code,)
        )

        if cursor.fetchone():

            cursor.execute("SELECT MAX(id) FROM students")

            max_id = cursor.fetchone()[0] or 1000

            data.student_code = (
                f"{data.student_code}-{max_id + 1}"
            )

        feature_dict = fill_features(data.dict())

        result = predict_single(feature_dict)

        cursor.execute(
            """
            INSERT INTO students (
                student_code,
                name,
                teacher_id,
                course,
                school,
                sex,
                age,
                studytime,
                absences,
                failures,
                schoolsup,
                famsup,
                paid,
                activities,
                higher,
                internet,
                health,
                goout,
                freetime,
                traveltime,
                risk_level,
                risk_probability,
                raw_features
            )
            VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                data.student_code,
                data.name,
                assigned_teacher_id,
                data.course,
                data.school,
                data.sex,
                data.age,
                data.studytime,
                data.absences,
                data.failures,
                data.schoolsup,
                data.famsup,
                data.paid,
                data.activities,
                data.higher,
                data.internet,
                data.health,
                data.goout,
                data.freetime,
                data.traveltime,
                result["risk_level"],
                result["risk_probability"],
                json.dumps(
                    feature_dict,
                    ensure_ascii=False
                )
            )
        )

        new_id = cursor.lastrowid

        if result["recommendations"]:

            recommendation = result["recommendations"][0]

            cursor.execute(
                """
                INSERT INTO interventions (
                    student_id,
                    title,
                    category,
                    status,
                    notes
                )
                VALUES (?, ?, ?, 'Pending', ?)
                """,
                (
                    new_id,
                    recommendation["action"],
                    recommendation["category"],
                    "توصية آلية فور تسجيل الطالب"
                )
            )

        conn.commit()

        return {
            "status": "success",
            "student_id": new_id,
            "student_code": data.student_code,
            "name": data.name,
            "risk_level": result["risk_level"],
            "risk_probability": result["risk_probability"],
            "recommendations": result["recommendations"],
            "message": f"تمت إضافة الطالب {data.name} بنجاح!"
        }

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


# ============================================================
# UPDATE STUDENT
# ============================================================

@app.put("/api/students/{student_id}")
def update_student(
    student_id: int,
    data: StudentInput,
    current_user: dict = Depends(get_current_user)
):

    conn = get_connection()

    try:
        cursor = conn.cursor()

        existing_student = ensure_student_access(
            cursor,
            student_id,
            current_user
        )

        # A teacher cannot change the student's course.
        if current_user["role"] == "teacher":

            validate_teacher_course(
                current_user["id"],
                data.course
            )

        else:

            # Admin can edit students, but the selected course
            # must still match the student's assigned teacher.
            validate_teacher_course(
                existing_student["teacher_id"],
                data.course
            )

        feature_dict = fill_features(data.dict())

        result = predict_single(feature_dict)

        cursor.execute(
            """
            UPDATE students
            SET
                name = ?,
                course = ?,
                age = ?,
                studytime = ?,
                absences = ?,
                failures = ?,
                schoolsup = ?,
                famsup = ?,
                higher = ?,
                goout = ?,
                health = ?,
                risk_level = ?,
                risk_probability = ?,
                raw_features = ?
            WHERE id = ?
            """,
            (
                data.name,
                data.course,
                data.age,
                data.studytime,
                data.absences,
                data.failures,
                data.schoolsup,
                data.famsup,
                data.higher,
                data.goout,
                data.health,
                result["risk_level"],
                result["risk_probability"],
                json.dumps(
                    feature_dict,
                    ensure_ascii=False
                ),
                student_id
            )
        )

        conn.commit()

        return {
            "status": "success",
            "student_id": student_id,
            "name": data.name,
            "risk_level": result["risk_level"],
            "risk_probability": result["risk_probability"],
            "message": "تم تحديث بيانات الطالب بنجاح!"
        }

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


# ============================================================
# DELETE STUDENT
# ============================================================

@app.delete("/api/students/{student_id}")
def delete_student(
    student_id: int,
    current_user: dict = Depends(get_current_user)
):

    conn = get_connection()

    try:
        cursor = conn.cursor()

        student = ensure_student_access(
            cursor,
            student_id,
            current_user
        )

        student_name = student["name"]

        cursor.execute(
            "DELETE FROM interventions WHERE student_id = ?",
            (student_id,)
        )

        cursor.execute(
            "DELETE FROM students WHERE id = ?",
            (student_id,)
        )

        conn.commit()

        return {
            "status": "success",
            "message": f"تم حذف الطالب '{student_name}' وسجلاته بنجاح!"
        }

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


# ============================================================
# ADD INTERVENTION
# ============================================================

@app.post("/api/interventions")
def add_intervention(
    item: InterventionCreate,
    current_user: dict = Depends(get_current_user)
):

    conn = get_connection()

    try:
        cursor = conn.cursor()

        ensure_student_access(
            cursor,
            item.student_id,
            current_user
        )

        cursor.execute(
            """
            INSERT INTO interventions (
                student_id,
                title,
                category,
                status,
                notes
            )
            VALUES (?, ?, ?, 'Pending', ?)
            """,
            (
                item.student_id,
                item.title,
                item.category,
                item.notes
            )
        )

        conn.commit()

        new_id = cursor.lastrowid

        return {
            "status": "success",
            "intervention_id": new_id,
            "message": "تم تسجيل خطة التدخل بنجاح"
        }

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


# ============================================================
# FRONTEND
# ============================================================

frontend_dir = os.path.join(BASE_DIR, "frontend")

if os.path.exists(frontend_dir):

    app.mount(
        "/static",
        StaticFiles(directory=frontend_dir),
        name="static"
    )


@app.get("/")
def serve_index():

    index_file = os.path.join(
        frontend_dir,
        "index.html"
    )

    if os.path.exists(index_file):
        return FileResponse(index_file)

    return {
        "message": "EduPulse API is running."
    }
