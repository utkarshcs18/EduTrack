import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

APP_NAME = "EduTrack"
DEFAULT_FILE = "studentData.csv"

REQUIRED_COLUMNS = [
    "Student_ID", "Name", "Gender", "Age", "Grade_Class",
    "Parental_Education", "Internet_Access", "Extracurricular_Activity",
    "Attendance_Percentage", "Study_Hours_Per_Week", "Math_Score",
    "Science_Score", "English_Score", "Final_Grade_Letter",
    "Enrollment_Date", "Remarks",
]

NUMERIC_COLUMNS = [
    "Age", "Attendance_Percentage", "Study_Hours_Per_Week",
    "Math_Score", "Science_Score", "English_Score",
]

SUBJECT_COLUMNS = ["Math_Score", "Science_Score", "English_Score"]
BOOL_COLUMNS = ["Internet_Access", "Extracurricular_Activity"]
GRADE_ORDER = ["O", "A", "B", "C", "D", "F"]
REMARK_MAP = {
    "O": "Outstanding",
    "A": "Excellent",
    "B": "Good",
    "C": "Average",
    "D": "Need Improvement",
    "F": "Failed",
}
PASSING_GRADES = {"O", "A", "B", "C", "D"}

# Standard score -> letter grade scale, used only to fill in a Final_Grade_Letter
# that is missing from the source data. Absent/missing subject scores count as 0
# when computing the average used for this lookup.
SCORE_TO_GRADE_SCALE = [(90, "O"), (80, "A"), (70, "B"), (60, "C"), (50, "D")]

def score_to_grade(avg_score):
    for threshold, grade in SCORE_TO_GRADE_SCALE:
        if avg_score >= threshold:
            return grade
    return "F"

def parse_args():
    parser = argparse.ArgumentParser(
        description="EduTrack - reusable student data analysis system."
    )
    parser.add_argument(
        "--file", default=DEFAULT_FILE,
        help=f"CSV file to analyze (default: {DEFAULT_FILE})",
    )
    return parser.parse_args()

def load_csv(file_path):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            f"Put a compatible CSV at '{DEFAULT_FILE}' or run with --file <path>."
        )

    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin1")

    df.columns = df.columns.astype(str).str.strip()
    return df

def validate_schema(df):
    return [col for col in REQUIRED_COLUMNS if col not in df.columns]

def clean_text(value):
    if pd.isna(value):
        return np.nan
    value = re.sub(r"\s+", " ", str(value).strip())
    if not value or value.upper() in {"NA", "N/A", "NONE", "NULL", "-"}:
        return np.nan
    return value

def title_case(value):
    if pd.isna(value):
        return np.nan
    return str(value).strip().title()

def roman_to_int(value):
    roman = str(value).strip().upper()
    if not roman:
        return np.nan
    if not re.fullmatch(r"[IVXLCDM]+", roman):
        return np.nan
    total = 0
    previous = 0
    for char in reversed(roman):
        current = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}[char]
        if current < previous:
            total -= current
        else:
            total += current
            previous = current
    return total

def normalize_grade_class(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if text in {"", "na", "n/a", "none", "null", "-"}:
        return np.nan

    match = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\b", text)
    if match:
        number = int(match.group(1))
    else:
        roman_match = re.search(r"\b([ivxlcdm]+)\b", text, flags=re.I)
        if not roman_match:
            return np.nan
        number = roman_to_int(roman_match.group(1))

    if pd.isna(number) or int(number) <= 0:
        return np.nan
    number = int(number)
    if 10 <= number % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix}"

def normalize_yes_no(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if text in {"yes", "y", "1", "true", "t"}:
        return "Yes"
    if text in {"no", "n", "0", "false", "f"}:
        return "No"
    return np.nan

def normalize_grade(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip().upper()
    mapping = {"O": "O", "A": "A", "B": "B", "C": "C", "D": "D", "F": "F"}
    return mapping.get(text, np.nan)

def clean_data(df):
    df = df.copy()
    report = {
        "rows_before": len(df),
        "exact_duplicates_removed": 0,
        "invalid_values": 0,
    }

    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].map(clean_text)

    if "Name" in df.columns:
        df["Name"] = df["Name"].map(title_case)

    if "Parental_Education" in df.columns:
        df["Parental_Education"] = df["Parental_Education"].map(title_case)
        df["Parental_Education"] = df["Parental_Education"].fillna("N/A")

    if "Gender" in df.columns:
        gender_map = {"m": "Male", "male": "Male", "f": "Female", "female": "Female"}
        before = df["Gender"].notna().sum()
        df["Gender"] = df["Gender"].map(
            lambda x: gender_map.get(str(x).strip().lower(), np.nan) if pd.notna(x) else np.nan
        )
        report["invalid_values"] += int(before - df["Gender"].notna().sum())

    if "Grade_Class" in df.columns:
        before = df["Grade_Class"].notna().sum()
        df["Grade_Class"] = df["Grade_Class"].map(normalize_grade_class)
        report["invalid_values"] += int(before - df["Grade_Class"].notna().sum())

    for col in BOOL_COLUMNS:
        if col in df.columns:
            before = df[col].notna().sum()
            df[col] = df[col].map(normalize_yes_no)
            report["invalid_values"] += int(before - df[col].notna().sum())
            df[col] = df[col].fillna("N/A")

    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            continue
        before = df[col].notna().sum()
        series = df[col].astype("string").str.replace("%", "", regex=False).str.strip()
        df[col] = pd.to_numeric(series, errors="coerce")
        report["invalid_values"] += int(before - df[col].notna().sum())

    if "Age" in df.columns:
        invalid = df["Age"].notna() & ((df["Age"] <= 0) | (df["Age"] % 1 != 0))
        report["invalid_values"] += int(invalid.sum())
        df.loc[invalid, "Age"] = np.nan
        df["Age"] = df["Age"].astype("Int64")

    if "Attendance_Percentage" in df.columns:
        invalid = df["Attendance_Percentage"].notna() & ~df["Attendance_Percentage"].between(0, 100)
        report["invalid_values"] += int(invalid.sum())
        df.loc[invalid, "Attendance_Percentage"] = np.nan
        df["Attendance_Percentage"] = df["Attendance_Percentage"].astype(float)
        df["Attendance_Percentage"] = df["Attendance_Percentage"].fillna(0.0)

    if "Study_Hours_Per_Week" in df.columns:
        invalid = df["Study_Hours_Per_Week"].notna() & ~df["Study_Hours_Per_Week"].between(0, 168)
        report["invalid_values"] += int(invalid.sum())
        df.loc[invalid, "Study_Hours_Per_Week"] = np.nan
        df["Study_Hours_Per_Week"] = df["Study_Hours_Per_Week"].astype(float)
        df["Study_Hours_Per_Week"] = df["Study_Hours_Per_Week"].fillna(0.0)

    for col in SUBJECT_COLUMNS:
        if col in df.columns:
            invalid = df[col].notna() & ~df[col].between(0, 100)
            report["invalid_values"] += int(invalid.sum())
            df.loc[invalid, col] = np.nan
            df[col] = df[col].astype(float)

    if "Final_Grade_Letter" in df.columns:
        before = df["Final_Grade_Letter"].notna().sum()
        df["Final_Grade_Letter"] = df["Final_Grade_Letter"].map(normalize_grade)
        report["invalid_values"] += int(before - df["Final_Grade_Letter"].notna().sum())

        # score (Math/Science/English), treating absent/missing subjects as 0.
        score_matrix = df[SUBJECT_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        avg_for_grade = score_matrix.mean(axis=1)
        computed_grade = avg_for_grade.apply(score_to_grade)
        report["grades_calculated_from_scores"] = int(df["Final_Grade_Letter"].isna().sum())
        df["Final_Grade_Letter"] = df["Final_Grade_Letter"].fillna(computed_grade)

        df["Final_Grade_Status"] = df["Final_Grade_Letter"].map(
            lambda x: "Pass" if x in PASSING_GRADES else ("Fail" if x == "F" else np.nan)
        )

        df["Remarks"] = df["Final_Grade_Letter"].map(REMARK_MAP)

        # the grade/status/remark are finalized using numeric scores,
        # replace remaining missing subject scores with "Absent" .
        for col in SUBJECT_COLUMNS:
            df[col] = df[col].apply(lambda v: "Absent" if pd.isna(v) else v)

    if "Enrollment_Date" in df.columns:
        df["Enrollment_Date"] = pd.to_datetime(
            df["Enrollment_Date"], errors="coerce", format="mixed", dayfirst=True
        )

    duplicates = int(df.duplicated().sum())
    if duplicates:
        df = df.drop_duplicates().reset_index(drop=True)
    report["exact_duplicates_removed"] = duplicates
    report["rows_after"] = len(df)
    report["missing_values_after_cleaning"] = int(df.isna().sum().sum())

    return df, report

def print_header(title):
    print("\n" + "=" * 64)
    print(title.center(64))
    print("=" * 64)

def display_value(value, date=False):
    if pd.isna(value):
        return "N/A"
    if date:
        return pd.Timestamp(value).strftime("%d-%m-%Y")
    if isinstance(value, (float, np.floating)):
        return f"{value:.2f}"
    return str(value)

def dataset_overview(df):
    print_header("DATASET OVERVIEW")
    print(f"Students/Rows     : {len(df)}")
    print(f"Attributes/Columns: {len(df.columns)}")
    print("\nColumns:")
    for i, col in enumerate(df.columns, 1):
        print(f"{i:>2}. {col}")
    print("\nData Types:")
    print(df.dtypes.to_string())

def dataset_health(raw_df, cleaned_df):
    print_header("DATASET HEALTH")
    print(f"Rows before cleaning          : {len(raw_df)}")
    print(f"Rows after cleaning           : {len(cleaned_df)}")
    print(f"Exact duplicates removed      : {raw_df.duplicated().sum()}")
    print(f"Missing values after cleaning : {int(cleaned_df.isna().sum().sum())}")

    duplicate_ids = cleaned_df["Student_ID"].duplicated(keep=False).sum()
    print(f"Rows with duplicate Student_ID: {duplicate_ids}")

    missing = cleaned_df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if not missing.empty:
        print("\nMissing values by column:")
        print(missing.to_string())

    print("\nImputed / placeholder values (not counted as missing above):")
    for col in ("Parental_Education", "Internet_Access", "Extracurricular_Activity"):
        if col in cleaned_df.columns:
            print(f"  {col:<26}: {int(cleaned_df[col].eq('N/A').sum())} marked 'N/A'")
    for col in ("Attendance_Percentage", "Study_Hours_Per_Week"):
        if col in cleaned_df.columns:
            print(f"  {col:<26}: {int(cleaned_df[col].eq(0).sum())} set to 0")
    for col in SUBJECT_COLUMNS:
        if col in cleaned_df.columns:
            print(f"  {col:<26}: {int(cleaned_df[col].eq('Absent').sum())} marked 'Absent'")

def validation_report(df, missing):
    print_header("DATASET VALIDATION")
    if not missing:
        print("File is readable")
        print("All required columns are present")
        print("Dataset is compatible with EduTrack")
        return True

    print("Dataset is missing required columns:")
    for col in missing:
        print(f"  - {col}")
    print("\nPlease provide a CSV that follows the EduTrack schema.")
    return False

def find_student(df):
    query = input("Enter Student ID: ").strip()
    return df[df["Student_ID"].astype(str).str.strip().str.upper() == query.upper()], query

def student_search(df):
    print_header("SEARCH STUDENT")
    result, query = find_student(df)
    if result.empty:
        print(f"\nStudent '{query}' was not found.")
        return
    for _, row in result.iterrows():
        print("\n" + "-" * 64)
        for col in REQUIRED_COLUMNS + ["Final_Grade_Status"]:
            if col not in row.index:
                continue
            value = display_value(row[col], col == "Enrollment_Date")
            print(f"{col.replace('_', ' '):<28}: {value}")

def performance_series(df):
    numeric = df[SUBJECT_COLUMNS].apply(pd.to_numeric, errors="coerce")
    return numeric.mean(axis=1, skipna=True)

def student_report(df):
    print_header("STUDENT REPORT")
    result, query = find_student(df)
    if result.empty:
        print(f"\nStudent '{query}' was not found.")
        return
    for _, row in result.iterrows():
        avg = performance_series(pd.DataFrame([row])).iloc[0]
        print(f"\nStudent ID       : {display_value(row['Student_ID'])}")
        print(f"Name             : {display_value(row['Name'])}")
        print(f"Gender           : {display_value(row['Gender'])}")
        print(f"Age              : {display_value(row['Age'])}")
        print(f"Grade Class      : {display_value(row['Grade_Class'])}")
        print(f"Parental Edu.    : {display_value(row['Parental_Education'])}")
        print(f"Internet Access  : {display_value(row['Internet_Access'])}")
        print(f"Extracurricular  : {display_value(row['Extracurricular_Activity'])}")
        print(f"Attendance       : {display_value(row['Attendance_Percentage'])}%")
        print(f"Study Hours/Week : {display_value(row['Study_Hours_Per_Week'])}")
        print(f"Math             : {display_value(row['Math_Score'])}/100")
        print(f"Science          : {display_value(row['Science_Score'])}/100")
        print(f"English          : {display_value(row['English_Score'])}/100")
        print(f"Subject Average  : {avg:.2f}/100" if pd.notna(avg) else "Subject Average  : N/A")
        print(f"Final Grade      : {display_value(row['Final_Grade_Letter'])}")
        print(f"Final Status     : {display_value(row['Final_Grade_Status'])}")
        print(f"Enrollment Date  : {display_value(row['Enrollment_Date'], True)}")
        print(f"Remarks          : {display_value(row['Remarks'])}")

def top_students(df):
    print_header("TOP STUDENTS")
    raw = input("How many top students? ").strip()
    try:
        n = int(raw)
        if n <= 0:
            raise ValueError
    except ValueError:
        print("Please enter a positive whole number.")
        return

    work = df.copy()
    work["Subject_Average"] = performance_series(work)
    work = work.dropna(subset=["Subject_Average"]).sort_values("Subject_Average", ascending=False).head(n)
    if work.empty:
        print("No students have enough subject scores for ranking.")
        return

    print(f"\n{'Rank':<6}{'Student ID':<14}{'Name':<24}{'Average':>10}")
    print("-" * 54)
    for rank, (_, row) in enumerate(work.iterrows(), 1):
        print(f"{rank:<6}{str(row['Student_ID']):<14}{str(row['Name'])[:22]:<24}{row['Subject_Average']:>10.2f}")

def class_statistics(df):
    print_header("CLASS STATISTICS")
    print(f"Students: {len(df)}")
    for col in SUBJECT_COLUMNS + ["Attendance_Percentage", "Study_Hours_Per_Week"]:
        series = pd.to_numeric(df[col], errors="coerce")
        suffix = "%" if col == "Attendance_Percentage" else ""
        print(f"{col.replace('_', ' '):<24}: Average={series.mean():.2f}{suffix} | Min={series.min():.2f}{suffix} | Max={series.max():.2f}{suffix}")
    avg = performance_series(df)
    print(f"\nOverall subject average: {avg.mean():.2f}/100")
    print(f"Pass rate: {df['Final_Grade_Status'].eq('Pass').mean() * 100:.2f}%")
    print(f"Fail rate: {df['Final_Grade_Status'].eq('Fail').mean() * 100:.2f}%")

def attendance_analysis(df):
    print_header("ATTENDANCE ANALYSIS")
    work = df.copy()
    work["Subject_Average"] = performance_series(work)
    work["Attendance_Group"] = pd.cut(
        work["Attendance_Percentage"],
        bins=[-np.inf, 74.999, 89.999, 100],
        labels=["Below 75%", "75-89%", "90-100%"],
    )
    result = work.groupby("Attendance_Group", observed=False).agg(
        Students=("Student_ID", "count"), Average_Score=("Subject_Average", "mean")
    )
    print(result.to_string(float_format=lambda x: f"{x:.2f}"))
    print("\nInterpretation: descriptive comparison only; it does not prove causation.")

def study_time_analysis(df):
    print_header("STUDY TIME ANALYSIS")
    work = df.copy()
    work["Subject_Average"] = performance_series(work)
    work["Study_Time_Group"] = pd.cut(
        work["Study_Hours_Per_Week"],
        bins=[-np.inf, 5, 10, 15, 168],
        labels=["0-5 hrs", "6-10 hrs", "11-15 hrs", "16+ hrs"],
    )
    result = work.groupby("Study_Time_Group", observed=False).agg(
        Students=("Student_ID", "count"), Average_Score=("Subject_Average", "mean")
    )
    print(result.to_string(float_format=lambda x: f"{x:.2f}"))
    print("\nInterpretation: descriptive comparison only; it does not prove causation.")

def grade_analysis(df):
    print_header("GRADE ANALYSIS")
    counts = df["Final_Grade_Letter"].value_counts(dropna=False)
    total = len(df)
    print("Final Grade Distribution:")
    for grade in GRADE_ORDER:
        count = int(counts.get(grade, 0))
        print(f"{grade:<6}: {count:>4} ({count / total * 100:.1f}%)")

    pass_count = int(df["Final_Grade_Status"].eq("Pass").sum())
    fail_count = int(df["Final_Grade_Status"].eq("Fail").sum())
    print(f"\nPass (O/A/B/C/D): {pass_count}")
    print(f"Fail (F)        : {fail_count}")

    missing = int(df["Final_Grade_Letter"].isna().sum())
    if missing:
        print(f"Unknown/Missing  : {missing}")

def full_report(df, cleaning_report):
    print_header("EDUTRACK FULL REPORT")
    print("DATASET")
    print("-" * 64)
    print(f"Students after cleaning : {len(df)}")
    print(f"Attributes              : {len(df.columns)}")
    print(f"Exact duplicates removed: {cleaning_report['exact_duplicates_removed']}")
    print(f"Invalid values handled  : {cleaning_report['invalid_values']}")

    print("\nACADEMIC PERFORMANCE")
    print("-" * 64)
    for col in SUBJECT_COLUMNS:
        numeric_col = pd.to_numeric(df[col], errors="coerce")
        absent = int(df[col].eq("Absent").sum())
        print(f"{col.replace('_', ' '):<24}: {numeric_col.mean():.2f}/100  (Absent: {absent})")
    print(f"{'Overall subject average':<24}: {performance_series(df).mean():.2f}/100")

    print("\nATTENDANCE / STUDY")
    print("-" * 64)
    print(f"{'Average attendance':<24}: {df['Attendance_Percentage'].mean():.2f}%")
    print(f"{'Average study hours':<24}: {df['Study_Hours_Per_Week'].mean():.2f}/week")

    print("\nFINAL STATUS")
    print("-" * 64)
    print(f"Pass (O/A/B/C/D): {df['Final_Grade_Status'].eq('Pass').sum()}")
    print(f"Fail (F)        : {df['Final_Grade_Status'].eq('Fail').sum()}")

    print("\nTOP 5 STUDENTS")
    print("-" * 64)
    top = df.assign(Subject_Average=performance_series(df)).dropna(subset=["Subject_Average"])
    top = top.sort_values("Subject_Average", ascending=False).head(5)
    for i, (_, row) in enumerate(top.iterrows(), 1):
        print(f"{i}. {row['Student_ID']} - {row['Name']} - {row['Subject_Average']:.2f}/100")

def save_cleaned_data(df, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "cleaned_student_data.csv"
    export = df.copy()
    if "Enrollment_Date" in export.columns:
        export["Enrollment_Date"] = export["Enrollment_Date"].dt.strftime("%d-%m-%Y")
    export.to_csv(output_file, index=False)
    return output_file

def save_summary(df, cleaning_report, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "summary.txt"
    lines = [
        f"students_after_cleaning: {len(df)}",
        f"attributes: {len(df.columns)}",
        f"exact_duplicates_removed: {cleaning_report['exact_duplicates_removed']}",
        f"invalid_values_handled: {cleaning_report['invalid_values']}",
        f"missing_values_after_cleaning: {int(df.isna().sum().sum())}",
        f"average_attendance: {df['Attendance_Percentage'].mean():.2f}%",
        f"average_study_hours_per_week: {df['Study_Hours_Per_Week'].mean():.2f}",
        f"average_math_score: {pd.to_numeric(df['Math_Score'], errors='coerce').mean():.2f}/100",
        f"average_science_score: {pd.to_numeric(df['Science_Score'], errors='coerce').mean():.2f}/100",
        f"average_english_score: {pd.to_numeric(df['English_Score'], errors='coerce').mean():.2f}/100",
        f"pass_count: {int(df['Final_Grade_Status'].eq('Pass').sum())}",
        f"fail_count: {int(df['Final_Grade_Status'].eq('Fail').sum())}",
        f"grades_calculated_from_scores: {cleaning_report.get('grades_calculated_from_scores', 0)}",
        f"absent_math_scores: {int(df['Math_Score'].eq('Absent').sum())}",
        f"absent_science_scores: {int(df['Science_Score'].eq('Absent').sum())}",
        f"absent_english_scores: {int(df['English_Score'].eq('Absent').sum())}",
    ]
    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_file

def main():
    args = parse_args()
    print("=" * 64)
    print(APP_NAME.center(64))
    print("Student Performance Analyzer".center(64))
    print("=" * 64)

    try:
        raw_df = load_csv(args.file)
    except (FileNotFoundError, pd.errors.ParserError, OSError) as exc:
        print(f"\nError: {exc}")
        return 1

    missing = validate_schema(raw_df)
    if not validation_report(raw_df, missing):
        return 1

    df, cleaning_report = clean_data(raw_df)
    output_dir = Path("output")
    cleaned_file = save_cleaned_data(df, output_dir)
    summary_file = save_summary(df, cleaning_report, output_dir)

    print_header("CLEANING COMPLETE")
    print(f"Rows before                 : {cleaning_report['rows_before']}")
    print(f"Rows after                  : {cleaning_report['rows_after']}")
    print(f"Exact duplicates removed    : {cleaning_report['exact_duplicates_removed']}")
    print(f"Invalid values handled      : {cleaning_report['invalid_values']}")
    print(f"Missing values after clean  : {cleaning_report['missing_values_after_cleaning']}")
    print(f"Cleaned dataset             : {cleaned_file}")
    print(f"Summary                     : {summary_file}")

    while True:
        print_header("MAIN MENU")
        print("1. Dataset Overview")
        print("2. Search Student")
        print("3. Student Report")
        print("4. Top Students")
        print("5. Class Statistics")
        print("6. Attendance Analysis")
        print("7. Study Time Analysis")
        print("8. Grade Analysis")
        print("9. Full Report")
        print("0. Exit")

        choice = input("\nEnter your choice: ").strip()
        if choice == "1":
            dataset_overview(df)
        elif choice == "2":
            student_search(df)
        elif choice == "3":
            student_report(df)
        elif choice == "4":
            top_students(df)
        elif choice == "5":
            class_statistics(df)
        elif choice == "6":
            attendance_analysis(df)
        elif choice == "7":
            study_time_analysis(df)
        elif choice == "8":
            grade_analysis(df)
        elif choice == "9":
            full_report(df, cleaning_report)
        elif choice == "0":
            print("\nEduTrack closed. Goodbye!")
            break
        else:
            print("\nInvalid choice. Please select 0-9.")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())