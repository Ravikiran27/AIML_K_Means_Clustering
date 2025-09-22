"""
Manual Data Analysis Implementation
No external libraries like sklearn, pandas, numpy - only built-in Python functions
"""

import csv
import math
import random
from collections import Counter


class ManualDataAnalysis:
    """
    A class implementing data analysis functions from scratch
    """
    
    def __init__(self):
        self.data = []
        self.headers = []
    
    def load_csv(self, filename):
        """Load CSV data manually"""
        with open(filename, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            self.headers = next(csv_reader)
            self.data = []
            
            for row in csv_reader:
                # Convert numeric columns to float where possible
                processed_row = []
                for i, value in enumerate(row):
                    try:
                        # Try to convert to float
                        processed_row.append(float(value))
                    except ValueError:
                        # Keep as string if not numeric
                        processed_row.append(value)
                self.data.append(processed_row)
        
        print(f"Loaded {len(self.data)} rows with {len(self.headers)} columns")
        print(f"Headers: {self.headers}")
        return self.data
    
    def get_column(self, column_name):
        """Extract a specific column by name"""
        if column_name not in self.headers:
            raise ValueError(f"Column '{column_name}' not found")
        
        column_index = self.headers.index(column_name)
        return [row[column_index] for row in self.data]
    
    def get_numeric_columns(self):
        """Get list of numeric column names"""
        numeric_cols = []
        for i, header in enumerate(self.headers):
            # Check if column contains numeric data
            sample_values = [row[i] for row in self.data[:10]]
            if all(isinstance(val, (int, float)) for val in sample_values):
                numeric_cols.append(header)
        return numeric_cols
    
    def basic_info(self):
        """Display basic information about the dataset"""
        print("\n=== DATASET INFORMATION ===")
        print(f"Shape: {len(self.data)} rows × {len(self.headers)} columns")
        print(f"Headers: {self.headers}")
        
        # Show first few rows
        print("\nFirst 5 rows:")
        for i in range(min(5, len(self.data))):
            row_dict = dict(zip(self.headers, self.data[i]))
            print(f"Row {i+1}: {row_dict}")
        
        # Identify data types
        print("\nColumn types:")
        for i, header in enumerate(self.headers):
            sample_val = self.data[0][i] if self.data else None
            col_type = type(sample_val).__name__
            print(f"  {header}: {col_type}")


class StatisticalFunctions:
    """
    Manual implementation of statistical functions
    """
    
    @staticmethod
    def mean(values):
        """Calculate arithmetic mean"""
        if not values:
            return None
        numeric_values = [v for v in values if isinstance(v, (int, float))]
        if not numeric_values:
            return None
        return sum(numeric_values) / len(numeric_values)
    
    @staticmethod
    def median(values):
        """Calculate median"""
        if not values:
            return None
        numeric_values = [v for v in values if isinstance(v, (int, float))]
        if not numeric_values:
            return None
        
        sorted_values = sorted(numeric_values)
        n = len(sorted_values)
        
        if n % 2 == 0:
            return (sorted_values[n//2 - 1] + sorted_values[n//2]) / 2
        else:
            return sorted_values[n//2]
    
    @staticmethod
    def mode(values):
        """Calculate mode (most frequent value)"""
        if not values:
            return None
        
        # Count frequencies
        freq_counter = Counter(values)
        max_freq = max(freq_counter.values())
        modes = [k for k, v in freq_counter.items() if v == max_freq]
        
        return modes[0] if len(modes) == 1 else modes
    
    @staticmethod
    def variance(values, sample=True):
        """Calculate variance"""
        if not values or len(values) < 2:
            return None
        
        numeric_values = [v for v in values if isinstance(v, (int, float))]
        if len(numeric_values) < 2:
            return None
        
        mean_val = StatisticalFunctions.mean(numeric_values)
        squared_diffs = [(x - mean_val) ** 2 for x in numeric_values]
        
        # Use sample variance (n-1) by default
        divisor = len(numeric_values) - 1 if sample else len(numeric_values)
        return sum(squared_diffs) / divisor
    
    @staticmethod
    def standard_deviation(values, sample=True):
        """Calculate standard deviation"""
        var = StatisticalFunctions.variance(values, sample)
        return math.sqrt(var) if var is not None else None
    
    @staticmethod
    def min_max(values):
        """Find minimum and maximum values"""
        if not values:
            return None, None
        
        numeric_values = [v for v in values if isinstance(v, (int, float))]
        if not numeric_values:
            return None, None
        
        return min(numeric_values), max(numeric_values)
    
    @staticmethod
    def quartiles(values):
        """Calculate Q1, Q2 (median), Q3"""
        if not values:
            return None, None, None
        
        numeric_values = [v for v in values if isinstance(v, (int, float))]
        if not numeric_values:
            return None, None, None
        
        sorted_values = sorted(numeric_values)
        n = len(sorted_values)
        
        # Q2 (median)
        q2 = StatisticalFunctions.median(sorted_values)
        
        # Q1 (median of lower half)
        lower_half = sorted_values[:n//2]
        q1 = StatisticalFunctions.median(lower_half)
        
        # Q3 (median of upper half)
        upper_half = sorted_values[(n+1)//2:] if n % 2 == 1 else sorted_values[n//2:]
        q3 = StatisticalFunctions.median(upper_half)
        
        return q1, q2, q3


def demonstrate_basic_analysis():
    """Demonstrate basic analysis on the air quality data"""
    
    # Initialize analyzer
    analyzer = ManualDataAnalysis()
    
    # Load data
    print("Loading air quality data...")
    analyzer.load_csv('s:\\AIML\\global_air_quality_data_10000.csv')
    
    # Show basic info
    analyzer.basic_info()
    
    # Get numeric columns for analysis
    numeric_cols = analyzer.get_numeric_columns()
    print(f"\nNumeric columns found: {numeric_cols}")
    
    # Analyze each numeric column
    print("\n=== STATISTICAL SUMMARY ===")
    stats = StatisticalFunctions()
    
    for col_name in numeric_cols[:5]:  # Analyze first 5 numeric columns
        values = analyzer.get_column(col_name)
        
        print(f"\n{col_name}:")
        print(f"  Mean: {stats.mean(values):.2f}")
        print(f"  Median: {stats.median(values):.2f}")
        print(f"  Std Dev: {stats.standard_deviation(values):.2f}")
        
        min_val, max_val = stats.min_max(values)
        print(f"  Min: {min_val:.2f}, Max: {max_val:.2f}")
        
        q1, q2, q3 = stats.quartiles(values)
        print(f"  Q1: {q1:.2f}, Q2: {q2:.2f}, Q3: {q3:.2f}")


if __name__ == "__main__":
    demonstrate_basic_analysis()