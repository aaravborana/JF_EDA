# JF_EDA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_attendance(present, t_days):
    if total_days <= 0:
        return "Total days cannot be zero or negative"
    if days_present < 0:
        return "Days present cannot be negative"
    if days_present > total_days:
        return "Days present cannot exceed total days"

    attendance_percentage = (present / t_days) * 100
    return f"Attendance Percentage: {attendance_percentage:.2f}%"


# Student 1 example :
days_present = 90
total_days = 100
print(calculate_attendance(days_present, total_days))


def extract_first_name(full_name):
    # Remove leading/trailing spaces and split the string
    name_parts = full_name.strip().split()

    if not name_parts:
        return "Empty string provided"

    return name_parts[0]


# Student 1 example :
student_name = "Kulesh_Kumar Umashankar Sen"
first_name = extract_first_name(student_name)
print(f"First name: {first_name}")

# Load the dataset into a Pandas DataFrame
def load_dataset(file_path):
    try:
        # Load the dataset from a CSV file
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")

# Perform basic exploratory data analysis (EDA)
def eda(df):
    # Display the first 5 rows of the dataset
    print("First 5 rows of the dataset:")
    print(df.head())

    # Check for null values
    print("\nNull values in the dataset:")
    print(df.isnull().sum())

    # Summarize data statistics
    print("\nData statistics:")
    print(df.describe())

# Load the dataset
file_path = "/content/learner_engagement_month_wise (2).csv"  # Replace with your dataset file path
df = load_dataset(file_path)

# Check if the dataset was loaded successfully before proceeding with EDA
if df is not None:
    # Perform EDA
    eda(df)
else:
    print(f"Error: Could not load the dataset from '{file_path}'. Please check the file path and permissions.")
    



# Load the dataset into a Pandas DataFrame
def load_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

# Filter records for a specific program
def filter_program(df, program_name):
    filtered_df = df[df['Program'] == program_name]
    return filtered_df

# Calculate total attendance hours for a given student
def total_attendance_hours(df, student_name):
    student_records = df[df['Student_Name'] == student_name]
    total_hours = student_records['Attendance_Hours'].sum()
    return total_hours



# Load the dataset into a Pandas DataFrame
def load_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

# Filter records for a specific program
def filter_program(df, program_name):
    filtered_df = df[df['Program'] == program_name]
    return filtered_df

#CL
from datetime import datetime, timedelta

class StudentAttendanceAnalyzer:
    def __init__(self, student_data):
        """
        Initialize with student attendance data
        student_data should be a DataFrame with columns:
        - date: Date of class
        - hours_present: Hours attended
        - subject: Subject name
        - is_holiday: Boolean indicating if it's a holiday
        """
        self.data = student_data
        self.clean_data()
        
    def clean_data(self):
        """Clean and prepare the data for analysis"""
        # Convert date to datetime
        self.data['date'] = pd.to_datetime(self.data['date'])
        # Fill missing values
        self.data['hours_present'].fillna(0, inplace=True)
        self.data['is_holiday'].fillna(False, inplace=True)
        
    def generate_attendance_report(self):
        """Generate comprehensive attendance report"""
        report = {
            'total_hours': self.calculate_total_hours(),
            'monthly_summary': self.get_monthly_summary(),
            'subject_wise_summary': self.get_subject_summary(),
            'attendance_trends': self.analyze_attendance_trends(),
            'attendance_stats': self.get_attendance_statistics()
        }
        return report
    
    def calculate_total_hours(self):
        """Calculate total hours attended"""
        return {
            'total_hours_present': self.data['hours_present'].sum(),
            'total_working_days': len(self.data[~self.data['is_holiday']]),
            'total_holidays': len(self.data[self.data['is_holiday']]),
            'average_daily_hours': self.data['hours_present'].mean()
        }
    
    def get_monthly_summary(self):
        """Generate monthly attendance summary"""
        monthly = self.data.groupby(self.data['date'].dt.strftime('%Y-%m'))[['hours_present']].agg({
            'hours_present': ['sum', 'mean', 'count']
        }).round(2)
        return monthly.to_dict()
    
    def get_subject_summary(self):
        """Generate subject-wise attendance summary"""
        return self.data.groupby('subject').agg({
            'hours_present': ['sum', 'mean', 'count']
        }).round(2).to_dict()
    
    def analyze_attendance_trends(self):
        """Analyze trends in attendance"""
        weekly_trend = self.data.groupby(self.data['date'].dt.day_name())['hours_present'].mean()
        return {
            'weekly_pattern': weekly_trend.to_dict(),
            'consecutive_absences': self.find_consecutive_absences(),
            'attendance_streak': self.find_longest_attendance_streak()
        }
    
    def get_attendance_statistics(self):
        """Calculate statistical measures of attendance"""
        return {
            'mean_hours': float(self.data['hours_present'].mean()),
            'median_hours': float(self.data['hours_present'].median()),
            'std_dev': float(self.data['hours_present'].std()),
            'max_hours': float(self.data['hours_present'].max()),
            'min_hours': float(self.data['hours_present'].min())
        }
    
    def find_consecutive_absences(self):
        """Find periods of consecutive absences"""
        absences = self.data[self.data['hours_present'] == 0]['date'].sort_values()
        if len(absences) == 0:
            return []
        
        gaps = []
        current_streak = [absences.iloc[0]]
        
        for i in range(1, len(absences)):
            if absences.iloc[i] - absences.iloc[i-1] == timedelta(days=1):
                current_streak.append(absences.iloc[i])
            else:
                if len(current_streak) > 1:
                    gaps.append({
                        'start': current_streak[0].strftime('%Y-%m-%d'),
                        'end': current_streak[-1].strftime('%Y-%m-%d'),
                        'days': len(current_streak)
                    })
                current_streak = [absences.iloc[i]]
                
        if len(current_streak) > 1:
            gaps.append({
                'start': current_streak[0].strftime('%Y-%m-%d'),
                'end': current_streak[-1].strftime('%Y-%m-%d'),
                'days': len(current_streak)
            })
            
        return gaps
    
    def find_longest_attendance_streak(self):
        """Find the longest streak of consecutive attendance"""
        attendance = self.data[self.data['hours_present'] > 0]['date'].sort_values()
        if len(attendance) == 0:
            return {'days': 0}
            
        max_streak = current_streak = 1
        streak_start = streak_end = attendance.iloc[0]
        current_start = attendance.iloc[0]
        
        for i in range(1, len(attendance)):
            if attendance.iloc[i] - attendance.iloc[i-1] == timedelta(days=1):
                current_streak += 1
                if current_streak > max_streak:
                    max_streak = current_streak
                    streak_start = current_start
                    streak_end = attendance.iloc[i]
            else:
                current_streak = 1
                current_start = attendance.iloc[i]
                
        return {
            'days': max_streak,
            'start_date': streak_start.strftime('%Y-%m-%d'),
            'end_date': streak_end.strftime('%Y-%m-%d')
        }

    def plot_attendance_trends(self):
        """Generate visualizations for attendance trends"""
        plt.figure(figsize=(15, 10))
        
        # Monthly trend
        plt.subplot(2, 2, 1)
        monthly_hours = self.data.groupby(self.data['date'].dt.strftime('%Y-%m'))['hours_present'].sum()
        monthly_hours.plot(kind='bar')
        plt.title('Monthly Attendance Hours')
        plt.xticks(rotation=45)
        
        # Subject-wise distribution
        plt.subplot(2, 2, 2)
        subject_hours = self.data.groupby('subject')['hours_present'].sum()
        subject_hours.plot(kind='pie', autopct='%1.1f%%')
        plt.title('Subject-wise Attendance Distribution')
        
        # Daily attendance pattern
        plt.subplot(2, 2, 3)
        daily_avg = self.data.groupby(self.data['date'].dt.day_name())['hours_present'].mean()
        daily_avg.plot(kind='bar')
        plt.title('Average Daily Attendance Pattern')
        plt.xticks(rotation=45)
        
        # Attendance distribution
        plt.subplot(2, 2, 4)
        sns.histplot(self.data['hours_present'])
        plt.title('Distribution of Daily Attendance Hours')
        
        plt.tight_layout()
        plt.show()

# Example usage
def create_sample_data():
    """Create sample data for demonstration"""
    dates = pd.date_range(start='2024-01-01', end='2024-03-31', freq='D')
    subjects = ['Math', 'Science', 'English', 'History']
    
    data = {
        'date': dates,
        'subject': [subjects[i % len(subjects)] for i in range(len(dates))],
        'hours_present': np.random.uniform(0, 8, len(dates)),
        'is_holiday': [i.weekday() >= 5 for i in dates]  # Weekends as holidays
    }
    
    return pd.DataFrame(data)

# Generate sample data and create report
sample_data = create_sample_data()
analyzer = StudentAttendanceAnalyzer(sample_data)
report = analyzer.generate_attendance_report()

# Print report
print("\nStudent Attendance Analysis Report")
print("=================================")
print("\n1. Overall Attendance Summary:")
print(f"Total Hours Present: {report['total_hours']['total_hours_present']:.2f}")
print(f"Total Working Days: {report['total_hours']['total_working_days']}")
print(f"Average Daily Hours: {report['total_hours']['average_daily_hours']:.2f}")

print("\n2. Attendance Statistics:")
print(f"Mean Hours: {report['attendance_stats']['mean_hours']:.2f}")
print(f"Median Hours: {report['attendance_stats']['median_hours']:.2f}")
print(f"Standard Deviation: {report['attendance_stats']['std_dev']:.2f}")

print("\n3. Longest Attendance Streak:")
streak = report['attendance_trends']['attendance_streak']
print(f"Duration: {streak['days']} days")
print(f"From: {streak['start_date']} to {streak['end_date']}")

# Generate visualizations
analyzer.plot_attendance_trends()

##Data Cleaning and Preparation

from datetime import datetime

def prepare_attendance_dataset(df):
    """
    Prepare the attendance dataset by cleaning and standardizing the data
    
    Parameters:
    df (pandas.DataFrame): Raw attendance dataset
    
    Returns:
    pandas.DataFrame: Cleaned and processed dataset
    """
    # Make a copy of the dataframe to avoid modifying the original
    df_clean = df.copy()
    
    # 1. Standardize column names
    def standardize_column_name(column):
        """Convert column names to snake_case format"""
        # Remove special characters and convert to lowercase
        clean_name = ''.join(e.lower() for e in column if e.isalnum() or e.isspace())
        # Replace spaces with underscores
        return clean_name.replace(' ', '_')
    
    df_clean.columns = [standardize_column_name(col) for col in df_clean.columns]
    
    # 2. Convert CreatedDate to datetime
    # First, handle various possible date formats
    def convert_to_datetime(date_str):
        """Convert various date string formats to datetime"""
        try:
            # Try parsing with pandas
            return pd.to_datetime(date_str)
        except:
            try:
                # Try common date formats
                date_formats = [
                    '%Y-%m-%d',
                    '%d-%m-%Y',
                    '%m/%d/%Y',
                    '%d/%m/%Y',
                    '%Y/%m/%d',
                    '%Y-%m-%d %H:%M:%S',
                    '%d-%m-%Y %H:%M:%S'
                ]
                
                for fmt in date_formats:
                    try:
                        return datetime.strptime(date_str, fmt)
                    except:
                        continue
                        
                raise ValueError(f"Unable to parse date: {date_str}")
            except Exception as e:
                print(f"Error converting date {date_str}: {str(e)}")
                return None
    
    # Convert createddate column
    date_col = 'createddate' if 'createddate' in df_clean.columns else 'created_date'
    df_clean[date_col] = df_clean[date_col].apply(convert_to_datetime)
    
    # 3. Add calculated columns
    def calculate_monthly_attendance(row):
        """Calculate monthly attendance rate"""
        if row['total_sessions'] == 0:
            return 0
        return (row['total_sessions_attended'] / row['total_sessions']) * 100
    
    # Ensure required columns exist and are numeric
    attendance_cols = ['total_sessions_attended', 'total_sessions']
    for col in attendance_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Calculate monthly attendance rate
    if all(col in df_clean.columns for col in attendance_cols):
        df_clean['monthly_attendance_rate'] = df_clean.apply(calculate_monthly_attendance, axis=1)
    
    # 4. Add additional useful features
    df_clean['month'] = df_clean[date_col].dt.month
    df_clean['year'] = df_clean[date_col].dt.year
    df_clean['day_of_week'] = df_clean[date_col].dt.day_name()
    
    # 5. Handle missing values
    df_clean = handle_missing_values(df_clean)
    
    return df_clean

def handle_missing_values(df):
    """
    Handle missing values in the dataset
    
    Parameters:
    df (pandas.DataFrame): Dataset with missing values
    
    Returns:
    pandas.DataFrame: Dataset with handled missing values
    """
    # Create a copy of the dataframe
    df_clean = df.copy()
    
    # Handle missing values based on data type
    for column in df_clean.columns:
        # Skip datetime columns
        if pd.api.types.is_datetime64_any_dtype(df_clean[column]):
            continue
            
        # For numeric columns
        if pd.api.types.is_numeric_dtype(df_clean[column]):
            # Fill missing values with median
            df_clean[column] = df_clean[column].fillna(df_clean[column].median())
            
        # For categorical/object columns
        else:
            # Fill missing values with mode
            df_clean[column] = df_clean[column].fillna(df_clean[column].mode()[0])
    
    return df_clean

def validate_dataset(df):
    """
    Validate the prepared dataset
    
    Parameters:
    df (pandas.DataFrame): Prepared dataset
    
    Returns:
    dict: Validation results
    """
    validation_results = {
        'total_rows': len(df),
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.to_dict(),
        'column_names': list(df.columns),
        'attendance_rate_range': {
            'min': df['monthly_attendance_rate'].min() if 'monthly_attendance_rate' in df.columns else None,
            'max': df['monthly_attendance_rate'].max() if 'monthly_attendance_rate' in df.columns else None
        }
    }
    
    return validation_results

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = pd.DataFrame({
        'CreatedDate': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'Student Name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
        'Total Sessions': [20, 20, 20],
        'Total Sessions Attended': [18, 15, 20]
    })
    
    # Prepare the dataset
    cleaned_data = prepare_attendance_dataset(sample_data)
    
    # Validate the results
    validation_results = validate_dataset(cleaned_data)
    
    # Print results
    print("\nCleaned Dataset Preview:")
    print(cleaned_data.head())
    print("\nValidation Results:")
    print(f"Total Rows: {validation_results['total_rows']}")
    print("\nColumn Data Types:")
    for col, dtype in validation_results['data_types'].items():
        print(f"{col}: {dtype}")
    print("\nAttendance Rate Range:")
    print(f"Min: {validation_results['attendance_rate_range']['min']:.2f}%")
    print(f"Max: {validation_results['attendance_rate_range']['max']:.2f}%")

#Creating a new column for "Cohort Year" from the "Cohort" column.

import re

def prepare_attendance_dataset(df):
    """
    Prepare the attendance dataset by cleaning and standardizing the data
    
    Parameters:
    df (pandas.DataFrame): Raw attendance dataset
    
    Returns:
    pandas.DataFrame: Cleaned and processed dataset
    """
    # Make a copy of the dataframe to avoid modifying the original
    df_clean = df.copy()
    
    # 1. Standardize column names
    def standardize_column_name(column):
        """Convert column names to snake_case format"""
        # Remove special characters and convert to lowercase
        clean_name = ''.join(e.lower() for e in column if e.isalnum() or e.isspace())
        # Replace spaces with underscores
        return clean_name.replace(' ', '_')
    
    df_clean.columns = [standardize_column_name(col) for col in df_clean.columns]
    
    # 2. Convert CreatedDate to datetime
    def convert_to_datetime(date_str):
        """Convert various date string formats to datetime"""
        try:
            return pd.to_datetime(date_str)
        except:
            try:
                date_formats = [
                    '%Y-%m-%d',
                    '%d-%m-%Y',
                    '%m/%d/%Y',
                    '%d/%m/%Y',
                    '%Y/%m/%d',
                    '%Y-%m-%d %H:%M:%S',
                    '%d-%m-%Y %H:%M:%S'
                ]
                
                for fmt in date_formats:
                    try:
                        return datetime.strptime(date_str, fmt)
                    except:
                        continue
                        
                raise ValueError(f"Unable to parse date: {date_str}")
            except Exception as e:
                print(f"Error converting date {date_str}: {str(e)}")
                return None
    
    # Convert createddate column
    date_col = 'createddate' if 'createddate' in df_clean.columns else 'created_date'
    df_clean[date_col] = df_clean[date_col].apply(convert_to_datetime)
    
    # 3. Extract CohortYear from Cohort column
    def extract_cohort_year(cohort_str):
        """
        Extract year from cohort string using various formats
        Examples: 'Cohort 2023', '2023 Batch', 'Batch_2023', etc.
        """
        if pd.isna(cohort_str):
            return None
            
        # Convert to string if not already
        cohort_str = str(cohort_str)
        
        # Try to find a four-digit year
        year_match = re.search(r'20\d{2}', cohort_str)
        if year_match:
            return int(year_match.group())
            
        # Try to find a two-digit year and convert to four digits
        short_year_match = re.search(r'\b\d{2}\b', cohort_str)
        if short_year_match:
            year = int(short_year_match.group())
            # Assume years 00-29 are 2000-2029, years 30-99 are 1930-1999
            return 2000 + year if year < 30 else 1900 + year
            
        return None

    # Add CohortYear column if Cohort column exists
    cohort_col = 'cohort' if 'cohort' in df_clean.columns else None
    if cohort_col:
        df_clean['cohort_year'] = df_clean[cohort_col].apply(extract_cohort_year)
        
        # Fill missing cohort years with the mode
        if df_clean['cohort_year'].notna().any():
            mode_year = df_clean['cohort_year'].mode()[0]
            df_clean['cohort_year'] = df_clean['cohort_year'].fillna(mode_year)
            
        # Convert to integer
        df_clean['cohort_year'] = df_clean['cohort_year'].astype('Int64')
    
    # 4. Add calculated columns
    def calculate_monthly_attendance(row):
        """Calculate monthly attendance rate"""
        if row['total_sessions'] == 0:
            return 0
        return (row['total_sessions_attended'] / row['total_sessions']) * 100
    
    # Ensure required columns exist and are numeric
    attendance_cols = ['total_sessions_attended', 'total_sessions']
    for col in attendance_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Calculate monthly attendance rate
    if all(col in df_clean.columns for col in attendance_cols):
        df_clean['monthly_attendance_rate'] = df_clean.apply(calculate_monthly_attendance, axis=1)
    
    # 5. Add additional useful features
    df_clean['month'] = df_clean[date_col].dt.month
    df_clean['year'] = df_clean[date_col].dt.year
    df_clean['day_of_week'] = df_clean[date_col].dt.day_name()
    
    # 6. Handle missing values
    df_clean = handle_missing_values(df_clean)
    
    return df_clean

def handle_missing_values(df):
    """Handle missing values in the dataset"""
    df_clean = df.copy()
    
    for column in df_clean.columns:
        if pd.api.types.is_datetime64_any_dtype(df_clean[column]):
            continue
            
        if pd.api.types.is_numeric_dtype(df_clean[column]):
            df_clean[column] = df_clean[column].fillna(df_clean[column].median())
        else:
            df_clean[column] = df_clean[column].fillna(df_clean[column].mode()[0])
    
    return df_clean

def validate_dataset(df):
    """Validate the prepared dataset"""
    validation_results = {
        'total_rows': len(df),
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.to_dict(),
        'column_names': list(df.columns),
        'attendance_rate_range': {
            'min': df['monthly_attendance_rate'].min() if 'monthly_attendance_rate' in df.columns else None,
            'max': df['monthly_attendance_rate'].max() if 'monthly_attendance_rate' in df.columns else None
        },
        'cohort_years': {
            'unique_values': sorted(df['cohort_year'].unique().tolist()) if 'cohort_year' in df.columns else None,
            'count': df['cohort_year'].value_counts().to_dict() if 'cohort_year' in df.columns else None
        }
    }
    
    return validation_results

# Example usage
if __name__ == "__main__":
    # Create sample data with cohort information
    sample_data = pd.DataFrame({
        'CreatedDate': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'Student Name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
        'Total Sessions': [20, 20, 20],
        'Total Sessions Attended': [18, 15, 20],
        'Cohort': ['Cohort 2023', 'Batch_2023', '2023 Spring']
    })
    
    # Prepare the dataset
    cleaned_data = prepare_attendance_dataset(sample_data)
    
    # Validate the results
    validation_results = validate_dataset(cleaned_data)
    
    # Print results
    print("\nCleaned Dataset Preview:")
    print(cleaned_data.head())
    print("\nValidation Results:")
    print(f"Total Rows: {validation_results['total_rows']}")
    print("\nColumn Data Types:")
    for col, dtype in validation_results['data_types'].items():
        print(f"{col}: {dtype}")
    print("\nCohort Years:")
    if validation_results['cohort_years']['unique_values']:
        print(f"Unique Years: {validation_results['cohort_years']['unique_values']}")
        print("\nCohort Year Distribution:")
        for year, count in validation_results['cohort_years']['count'].items():
            print(f"{year}: {count} students")

##Data Visualization
import matplotlib.pyplot as plt
!pip install seaborn
import seaborn as sns
import warnings

# Filter out warnings related to missing style sheets
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


def create_attendance_visualizations(df):
    """
    Create comprehensive visualizations for attendance data
    
    Parameters:
    df (pandas.DataFrame): Cleaned attendance dataset with columns:
        - created_date or createddate
        - total_sessions_attended
        - monthly_attendance_rate
    """
    # Set the style for better-looking graphs
  
    
    # Create a figure with subplots
    fig = plt.figure(figsize=(15, 12))
    
    # 1. Bar Chart - Total Sessions Attended by Month
    plt.subplot(3, 1, 1)
    
    # Group by month and sum the sessions
    date_col = 'created_date' if 'created_date' in df.columns else 'createddate'
    monthly_sessions = df.groupby(df[date_col].dt.strftime('%Y-%m'))['total_sessions_attended'].sum()
    
    # Create bar plot
    ax1 = monthly_sessions.plot(kind='bar', color='skyblue')
    plt.title('Total Sessions Attended by Month', pad=20, fontsize=14)
    plt.xlabel('Month')
    plt.ylabel('Total Sessions Attended')
    plt.xticks(rotation=45)
    
    # Add value labels on top of bars
    for i, v in enumerate(monthly_sessions):
        ax1.text(i, v, str(int(v)), ha='center', va='bottom')
    
    # 2. Histogram - Attendance Percentage Distribution
    plt.subplot(3, 1, 2)
    
    # Create histogram
    sns.histplot(data=df, x='monthly_attendance_rate', bins=20, color='lightgreen')
    plt.title('Distribution of Monthly Attendance Rates', pad=20, fontsize=12)
    plt.xlabel('Attendance Rate (%)')
    plt.ylabel('Frequency')
    
    # Add mean and median lines
    mean_attendance = df['monthly_attendance_rate'].mean()
    median_attendance = df['monthly_attendance_rate'].median()
    
    plt.axvline(mean_attendance, color='red', linestyle='--', label=f'Mean: {mean_attendance:.1f}%')
    plt.axvline(median_attendance, color='green', linestyle='--', label=f'Median: {median_attendance:.1f}%')
    plt.legend()
    
    # 3. Line Graph - Attendance Trends
    plt.subplot(3, 1, 3)
    
    # Calculate average attendance rate by date
    daily_attendance = df.groupby(date_col)['monthly_attendance_rate'].mean()
    
    # Create line plot
    plt.plot(daily_attendance.index, daily_attendance.values, marker='o', linewidth=2, markersize=6)
    plt.title('Attendance Rate Trends Over Time', pad=20, fontsize=12)
    plt.xlabel('Date')
    plt.ylabel('Average Attendance Rate (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def create_detailed_monthly_analysis(df):
    """
    Create additional monthly analysis visualizations
    
    Parameters:
    df (pandas.DataFrame): Cleaned attendance dataset
    """
    date_col = 'created_date' if 'created_date' in df.columns else 'createddate'
    
    # Create figure for monthly analysis
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Box Plot - Monthly Attendance Distribution
    plt.subplot(2, 1, 1)
    df['month'] = df[date_col].dt.strftime('%Y-%m')
    sns.boxplot(data=df, x='month', y='monthly_attendance_rate')
    plt.title('Monthly Attendance Rate Distribution', pad=20, fontsize=12)
    plt.xlabel('Month')
    plt.ylabel('Attendance Rate (%)')
    plt.xticks(rotation=45)
    
    # 2. Violin Plot - Detailed Distribution by Month
    plt.subplot(2, 1, 2)
    sns.violinplot(data=df, x='month', y='monthly_attendance_rate')
    plt.title('Detailed Monthly Attendance Distribution', pad=20, fontsize=12)
    plt.xlabel('Month')
    plt.ylabel('Attendance Rate (%)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    return fig

# Example usage with sample data
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range(start='2024-01-01', end='2024-03-31', freq='D')
    sample_data = pd.DataFrame({
        'created_date': dates,
        'total_sessions_attended': np.random.randint(0, 10, size=len(dates)),
        'monthly_attendance_rate': np.random.uniform(60, 100, size=len(dates))
    })
    
    # Create visualizations
    attendance_fig = create_attendance_visualizations(sample_data)
    monthly_fig = create_detailed_monthly_analysis(sample_data)
    
    # Display plots
    plt.show()
    
    # Print summary statistics
    print("\nAttendance Summary Statistics:")
    print(f"Average Attendance Rate: {sample_data['monthly_attendance_rate'].mean():.2f}%")
    print(f"Median Attendance Rate: {sample_data['monthly_attendance_rate'].median():.2f}%")
    print(f"Total Sessions Attended: {sample_data['total_sessions_attended'].sum()}")

## Insights and Analytics


class StudentEngagementAnalyzer:
    def __init__(self, df):
        """
        Initialize with attendance data
        
        Parameters:
        df (pandas.DataFrame): Dataset with columns for student_id, subject,
            attendance_rate, created_date/createddate
        """
        self.df = df.copy()
        self.date_col = 'created_date' if 'created_date' in self.df.columns else 'createddate'
    
    def identify_low_attendance_students(self, threshold=50):
        """
        Identify students with attendance rates below threshold
        
        Parameters:
        threshold (float): Attendance rate threshold (default: 50%)
        
        Returns:
        pandas.DataFrame: Students with low attendance
        """
        low_attendance = self.df.groupby('student_id').agg({
            'monthly_attendance_rate': 'mean',
            'total_sessions_attended': 'sum',
            'total_sessions': 'sum'
        }).round(2)
        
        low_attendance = low_attendance[low_attendance['monthly_attendance_rate'] < threshold]
        low_attendance['attendance_status'] = 'At Risk'
        
        # Add trend analysis
        low_attendance['missed_sessions'] = (
            low_attendance['total_sessions'] - low_attendance['total_sessions_attended']
        )
        
        return low_attendance.sort_values('monthly_attendance_rate')
    
    def analyze_subject_engagement(self):
        """
        Analyze engagement metrics by subject
        
        Returns:
        tuple: (DataFrame with subject metrics, matplotlib figure)
        """
        subject_metrics = self.df.groupby('subject').agg({
            'monthly_attendance_rate': ['mean', 'median', 'std'],
            'total_sessions_attended': 'sum',
            'student_id': 'nunique'
        }).round(2)
        
        subject_metrics.columns = [
            'avg_attendance_rate', 'median_attendance_rate', 
            'attendance_std', 'total_sessions', 'unique_students'
        ]
        
        # Calculate engagement score (weighted average of metrics)
        subject_metrics['engagement_score'] = (
            0.4 * subject_metrics['avg_attendance_rate'] +
            0.3 * subject_metrics['median_attendance_rate'] +
            0.2 * (100 - subject_metrics['attendance_std']) +
            0.1 * (subject_metrics['unique_students'] / subject_metrics['unique_students'].max() * 100)
        ).round(2)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot average attendance rates
        subject_metrics['avg_attendance_rate'].sort_values().plot(
            kind='bar', ax=ax1, color='skyblue'
        )
        ax1.set_title('Average Attendance Rate by Subject')
        ax1.set_ylabel('Attendance Rate (%)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot engagement scores
        subject_metrics['engagement_score'].sort_values().plot(
            kind='bar', ax=ax2, color='lightgreen'
        )
        ax2.set_title('Overall Engagement Score by Subject')
        ax2.set_ylabel('Engagement Score')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        return subject_metrics.sort_values('engagement_score', ascending=False), fig
    
    def analyze_monthly_trends(self):
        """
        Analyze monthly engagement trends
        
        Returns:
        tuple: (DataFrame with monthly metrics, matplotlib figure)
        """
        # Extract month from date column
        self.df['month'] = pd.to_datetime(self.df[self.date_col]).dt.strftime('%Y-%m')
        
        monthly_metrics = self.df.groupby('month').agg({
            'monthly_attendance_rate': ['mean', 'median', 'std'],
            'total_sessions_attended': 'sum',
            'student_id': 'nunique'
        }).round(2)
        
        monthly_metrics.columns = [
            'avg_attendance_rate', 'median_attendance_rate',
            'attendance_std', 'total_sessions', 'active_students'
        ]
        
        # Calculate month-over-month changes
        monthly_metrics['attendance_change'] = monthly_metrics['avg_attendance_rate'].pct_change() * 100
        
        # Create visualization
        fig, axes = plt.subplots(3, 1, figsize=(12, 15))
        
        # Plot 1: Attendance Rates Over Time
        monthly_metrics['avg_attendance_rate'].plot(
            marker='o', ax=axes[0], color='blue', linewidth=2
        )
        monthly_metrics['median_attendance_rate'].plot(
            marker='s', ax=axes[0], color='green', linewidth=2
        )
        axes[0].set_title('Attendance Rates Over Time')
        axes[0].set_ylabel('Attendance Rate (%)')
        axes[0].legend(['Average', 'Median'])
        axes[0].grid(True)
        
        # Plot 2: Active Students Over Time
        monthly_metrics['active_students'].plot(
            kind='bar', ax=axes[1], color='orange'
        )
        axes[1].set_title('Active Students by Month')
        axes[1].set_ylabel('Number of Students')
        axes[1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Month-over-Month Changes
        monthly_metrics['attendance_change'].plot(
            kind='bar', ax=axes[2], color='red'
        )
        axes[2].set_title('Month-over-Month Attendance Rate Changes')
        axes[2].set_ylabel('Change (%)')
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        
        return monthly_metrics, fig
    
    def generate_summary_report(self):
        """
        Generate a comprehensive summary report
        
        Returns:
        dict: Summary metrics and insights
        """
        summary = {
            'overall_metrics': {
                'average_attendance_rate': self.df['monthly_attendance_rate'].mean(),
                'median_attendance_rate': self.df['monthly_attendance_rate'].median(),
                'total_students': self.df['student_id'].nunique(),
                'total_sessions': self.df['total_sessions'].sum()
            },
            'low_attendance_summary': {
                'count': len(self.identify_low_attendance_students()),
                'average_rate': self.identify_low_attendance_students()['monthly_attendance_rate'].mean()
            },
            'subject_summary': self.analyze_subject_engagement()[0].to_dict(),
            'monthly_summary': self.analyze_monthly_trends()[0].to_dict()
        }
        
        return summary

# Example usage
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range(start='2024-01-01', end='2024-03-31', freq='D')
    subjects = ['Math', 'Science', 'English', 'History']
    student_ids = range(1, 51)  # 50 students
    
    sample_data = pd.DataFrame({
        'created_date': np.repeat(dates, len(student_ids)),
        'student_id': np.tile(student_ids, len(dates)),
        'subject': np.random.choice(subjects, len(dates) * len(student_ids)),
        'monthly_attendance_rate': np.random.uniform(30, 100, len(dates) * len(student_ids)),
        'total_sessions': np.random.randint(15, 25, len(dates) * len(student_ids)),
        'total_sessions_attended': np.random.randint(10, 20, len(dates) * len(student_ids))
    })
    
    # Create analyzer instance
    analyzer = StudentEngagementAnalyzer(sample_data)
    
    # Generate reports
    print("\nLow Attendance Students:")
    print(analyzer.identify_low_attendance_students().head())
    
    print("\nSubject Engagement Analysis:")
    subject_metrics, _ = analyzer.analyze_subject_engagement()
    print(subject_metrics)
    
    print("\nMonthly Trends:")
    monthly_metrics, _ = analyzer.analyze_monthly_trends()
    print(monthly_metrics)
    
    plt.show()

##identifying students with over 90% attendence rate 

# Reload the data to ensure the variable is properly defined
data = pd.read_csv("/learner_engagement_month_wise (2).csv")

# Re-calculate the attendance percentage and filter the data
data['AttendancePercentage'] = (data['TotalSessionAttended'] / data['TotalSessions']) * 100
students_90_percent_attendance = data[data['AttendancePercentage'] >= 90]

# Define the columns you want to include in the report
report_columns = ['Sapid', 'FirstName', 'MiddleName', 'LastName', 'Program', 'AttendancePercentage'] # Replace with your desired columns

# Select relevant columns for the report
report = students_90_percent_attendance[report_columns]

# Display the filtered report again
report.head()

##Report Generation

!pip install reportlab
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from io import BytesIO

class AttendanceReportGenerator:
    def __init__(self, df):
        self.df = df
        self.styles = getSampleStyleSheet()
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30
        )
        
    def generate_visualizations(self):
        """Generate all required visualizations and save them"""
        # Create figure for attendance distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.df, x='monthly_attendance_rate', bins=30)
        plt.title('Distribution of Monthly Attendance Rates')
        plt.xlabel('Attendance Rate (%)')
        plt.ylabel('Frequency')
        
        # Save plot to BytesIO object
        attendance_dist = BytesIO()
        plt.savefig(attendance_dist, format='png', bbox_inches='tight')
        plt.close()
        
        # Create monthly trends plot
        plt.figure(figsize=(10, 6))
        monthly_avg = self.df.groupby('month')['monthly_attendance_rate'].mean()
        monthly_avg.plot(kind='line', marker='o')
        plt.title('Monthly Average Attendance Trends')
        plt.xlabel('Month')
        plt.ylabel('Average Attendance Rate (%)')
        plt.xticks(rotation=45)
        
        # Save plot to BytesIO object
        monthly_trends = BytesIO()
        plt.savefig(monthly_trends, format='png', bbox_inches='tight')
        plt.close()
        
        return attendance_dist, monthly_trends
    
    def generate_summary_statistics(self):
        """Generate summary statistics for the report"""
        stats = {
            'Total Students': self.df['student_id'].nunique(),
            'Average Attendance Rate': f"{self.df['monthly_attendance_rate'].mean():.2f}%",
            'Median Attendance Rate': f"{self.df['monthly_attendance_rate'].median():.2f}%",
            'Students Below 75%': len(self.df[self.df['monthly_attendance_rate'] < 75]['student_id'].unique()),
            'Students Above 90%': len(self.df[self.df['monthly_attendance_rate'] > 90]['student_id'].unique())
        }
        return stats
    
    def create_pdf_report(self, output_path='attendance_report.pdf'):
        """Create the final PDF report"""
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        story = []
        
        # Add title
        story.append(Paragraph('Student Attendance Analysis Report', self.title_style))
        story.append(Spacer(1, 20))
        
        # Add summary statistics
        stats = self.generate_summary_statistics()
        stats_table_data = [[key, value] for key, value in stats.items()]
        stats_table = Table(stats_table_data)
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 14),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(stats_table)
        story.append(Spacer(1, 30))
        
        # Add visualizations
        attendance_dist, monthly_trends = self.generate_visualizations()
        
        # Add distribution plot
        story.append(Paragraph('Attendance Rate Distribution', self.styles['Heading2']))
        story.append(Image(attendance_dist, width=400, height=300))
        story.append(Spacer(1, 20))
        
        # Add trends plot
        story.append(Paragraph('Monthly Attendance Trends', self.styles['Heading2']))
        story.append(Image(monthly_trends, width=400, height=300))
        
        # Generate PDF
        doc.build(story)
        print(f"Report generated successfully: {output_path}")

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = pd.DataFrame({
        'student_id': range(1, 101),
        'monthly_attendance_rate': np.random.normal(80, 10, 100),
        'month': np.random.choice(pd.date_range('2024-01-01', '2024-12-31', freq='M'), 100)
    })
    
    # Generate report
    report_generator = AttendanceReportGenerator(sample_data)
    report_generator.create_pdf_report()




 

   






