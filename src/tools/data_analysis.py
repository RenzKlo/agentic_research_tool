"""
Data Analysis Tool

This tool provides comprehensive data analysis capabilities including
statistical analysis, data visualization, and data manipulation.
"""

import logging
import io
import base64
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class DataAnalysisInput(BaseModel):
    """Input schema for data analysis operations"""
    operation: str = Field(description="Analysis operation: load, describe, visualize, correlate, cluster, pca, statistics")
    data_source: Optional[str] = Field(None, description="Path to data file or variable name")
    data_content: Optional[str] = Field(None, description="CSV content as string")
    columns: Optional[List[str]] = Field(None, description="Specific columns to analyze")
    chart_type: Optional[str] = Field(None, description="Type of chart: histogram, scatter, box, heatmap, bar")
    target_column: Optional[str] = Field(None, description="Target column for analysis")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Additional parameters")

class DataAnalysisTool(BaseTool):
    """Tool for comprehensive data analysis"""
    
    name: str = "data_analysis"
    description: str = """
    Perform comprehensive data analysis including:
    - load: Load data from file or CSV string
    - describe: Generate descriptive statistics
    - visualize: Create charts and plots (histogram, scatter, box, heatmap, bar)
    - correlate: Calculate correlations between variables
    - cluster: Perform K-means clustering
    - pca: Principal component analysis
    - statistics: Advanced statistical tests and analysis
    
    Use this tool for any data analysis, visualization, or statistical tasks.
    """
    args_schema = DataAnalysisInput
    
    # Class-level data store
    _data_store = {}
    
    def __init__(self):
        super().__init__()
        # Use class-level data store instead of instance attribute
    
    def _run(self,
             operation: str,
             data_source: Optional[str] = None,
             data_content: Optional[str] = None,
             columns: Optional[List[str]] = None,
             chart_type: Optional[str] = None,
             target_column: Optional[str] = None,
             parameters: Optional[Dict[str, Any]] = None) -> str:
        """Execute data analysis operation"""
        try:
            if operation == "load":
                return self._load_data(data_source, data_content)
            elif operation == "describe":
                return self._describe_data(data_source, columns)
            elif operation == "visualize":
                return self._create_visualization(data_source, chart_type, columns, target_column, parameters)
            elif operation == "correlate":
                return self._correlation_analysis(data_source, columns)
            elif operation == "cluster":
                return self._clustering_analysis(data_source, columns, parameters)
            elif operation == "pca":
                return self._pca_analysis(data_source, columns, parameters)
            elif operation == "statistics":
                return self._statistical_analysis(data_source, columns, parameters)
            else:
                return f"Unknown operation: {operation}"
                
        except Exception as e:
            error_msg = f"Data analysis error: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def _load_data(self, data_source: Optional[str], data_content: Optional[str]) -> str:
        """Load data from file or string"""
        try:
            if data_content:
                # Load from CSV string
                df = pd.read_csv(io.StringIO(data_content))
                dataset_name = "csv_data"
            elif data_source:
                # Load from file
                if data_source.endswith('.csv'):
                    df = pd.read_csv(data_source)
                elif data_source.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(data_source)
                elif data_source.endswith('.json'):
                    df = pd.read_json(data_source)
                else:
                    return f"Unsupported file format: {data_source}"
                dataset_name = data_source.split('/')[-1].split('.')[0]
            else:
                return "No data source provided"
            
            # Store the dataset
            self._data_store[dataset_name] = df
            
            # Basic info about the dataset
            info = [
                f"Successfully loaded dataset: {dataset_name}",
                f"Shape: {df.shape}",
                f"Columns: {list(df.columns)}",
                f"Data types:\n{df.dtypes.to_string()}",
                f"Missing values:\n{df.isnull().sum().to_string()}",
                f"First 5 rows:\n{df.head().to_string()}"
            ]
            
            return "\n\n".join(info)
            
        except Exception as e:
            return f"Error loading data: {str(e)}"
    
    def _describe_data(self, data_source: str, columns: Optional[List[str]]) -> str:
        """Generate descriptive statistics"""
        try:
            df = self._get_dataset(data_source)
            if isinstance(df, str):  # Error message
                return df
            
            if columns:
                df = df[columns]
            
            # Separate numeric and categorical columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(include=['object']).columns
            
            analysis = []
            
            # Basic statistics
            analysis.append("DATASET OVERVIEW")
            analysis.append(f"Shape: {df.shape}")
            analysis.append(f"Memory usage: {df.memory_usage(deep=True).sum()} bytes")
            
            # Numeric statistics
            if len(numeric_cols) > 0:
                analysis.append("\nNUMERIC COLUMNS STATISTICS")
                analysis.append(df[numeric_cols].describe().to_string())
                
                # Skewness and kurtosis
                skew_kurt = pd.DataFrame({
                    'Skewness': df[numeric_cols].skew(),
                    'Kurtosis': df[numeric_cols].kurtosis()
                })
                analysis.append(f"\nSkewness and Kurtosis:\n{skew_kurt.to_string()}")
            
            # Categorical statistics
            if len(categorical_cols) > 0:
                analysis.append("\nCATEGORICAL COLUMNS STATISTICS")
                for col in categorical_cols:
                    analysis.append(f"\n{col}:")
                    analysis.append(f"  Unique values: {df[col].nunique()}")
                    analysis.append(f"  Most frequent: {df[col].mode().iloc[0] if not df[col].mode().empty else 'N/A'}")
                    analysis.append(f"  Top 5 values:\n{df[col].value_counts().head().to_string()}")
            
            # Missing values analysis
            missing = df.isnull().sum()
            if missing.sum() > 0:
                missing_pct = (missing / len(df)) * 100
                missing_df = pd.DataFrame({
                    'Missing Count': missing,
                    'Missing Percentage': missing_pct
                }).query('`Missing Count` > 0')
                analysis.append(f"\nMISSING VALUES:\n{missing_df.to_string()}")
            
            return "\n".join(analysis)
            
        except Exception as e:
            return f"Error describing data: {str(e)}"
    
    def _create_visualization(self, data_source: str, chart_type: str, columns: Optional[List[str]], 
                            target_column: Optional[str], parameters: Optional[Dict[str, Any]]) -> str:
        """Create data visualizations"""
        try:
            df = self._get_dataset(data_source)
            if isinstance(df, str):  # Error message
                return df
            
            if columns:
                df_viz = df[columns]
            else:
                df_viz = df
            
            # Set matplotlib style
            plt.style.use('default')
            
            if chart_type == "histogram":
                return self._create_histogram(df_viz, target_column)
            elif chart_type == "scatter":
                return self._create_scatter_plot(df_viz, columns, parameters)
            elif chart_type == "box":
                return self._create_box_plot(df_viz, target_column)
            elif chart_type == "heatmap":
                return self._create_heatmap(df_viz)
            elif chart_type == "bar":
                return self._create_bar_chart(df_viz, target_column)
            else:
                return self._auto_visualize(df_viz)
                
        except Exception as e:
            return f"Error creating visualization: {str(e)}"
    
    def _create_histogram(self, df: pd.DataFrame, target_column: Optional[str]) -> str:
        """Create histogram"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if target_column and target_column in numeric_cols:
            cols_to_plot = [target_column]
        else:
            cols_to_plot = numeric_cols[:4]  # Plot first 4 numeric columns
        
        if len(cols_to_plot) == 0:
            return "No numeric columns found for histogram"
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, col in enumerate(cols_to_plot):
            if i < 4:
                axes[i].hist(df[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'Distribution of {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
        
        # Hide unused subplots
        for i in range(len(cols_to_plot), 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot
        img_path = f"histogram_{data_source}_{target_column or 'all'}.png"
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return f"Created histogram plot saved as {img_path}. Analyzed columns: {list(cols_to_plot)}"
    
    def _create_scatter_plot(self, df: pd.DataFrame, columns: Optional[List[str]], 
                           parameters: Optional[Dict[str, Any]]) -> str:
        """Create scatter plot"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return "Need at least 2 numeric columns for scatter plot"
        
        x_col = columns[0] if columns and len(columns) > 0 else numeric_cols[0]
        y_col = columns[1] if columns and len(columns) > 1 else numeric_cols[1]
        
        plt.figure(figsize=(10, 8))
        plt.scatter(df[x_col], df[y_col], alpha=0.6)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f'Scatter Plot: {x_col} vs {y_col}')
        plt.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        corr = df[x_col].corr(df[y_col])
        plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=plt.gca().transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        img_path = f"scatter_{x_col}_vs_{y_col}.png"
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return f"Created scatter plot saved as {img_path}. Correlation between {x_col} and {y_col}: {corr:.3f}"
    
    def _create_box_plot(self, df: pd.DataFrame, target_column: Optional[str]) -> str:
        """Create box plot"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if target_column and target_column in numeric_cols:
            cols_to_plot = [target_column]
        else:
            cols_to_plot = numeric_cols[:4]
        
        if len(cols_to_plot) == 0:
            return "No numeric columns found for box plot"
        
        plt.figure(figsize=(12, 8))
        df[cols_to_plot].boxplot()
        plt.title('Box Plot Distribution')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        img_path = f"boxplot_{target_column or 'all'}.png"
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return f"Created box plot saved as {img_path}. Analyzed columns: {list(cols_to_plot)}"
    
    def _create_heatmap(self, df: pd.DataFrame) -> str:
        """Create correlation heatmap"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return "Need at least 2 numeric columns for correlation heatmap"
        
        correlation_matrix = df[numeric_cols].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                    square=True, linewidths=0.5)
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        
        img_path = "correlation_heatmap.png"
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Find strong correlations
        strong_corr = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    strong_corr.append(f"{correlation_matrix.columns[i]} - {correlation_matrix.columns[j]}: {corr_val:.3f}")
        
        result = f"Created correlation heatmap saved as {img_path}."
        if strong_corr:
            result += f"\nStrong correlations (|r| > 0.5):\n" + "\n".join(strong_corr)
        
        return result
    
    def _create_bar_chart(self, df: pd.DataFrame, target_column: Optional[str]) -> str:
        """Create bar chart for categorical data"""
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        if target_column and target_column in categorical_cols:
            col_to_plot = target_column
        elif len(categorical_cols) > 0:
            col_to_plot = categorical_cols[0]
        else:
            return "No categorical columns found for bar chart"
        
        value_counts = df[col_to_plot].value_counts().head(10)
        
        plt.figure(figsize=(12, 8))
        value_counts.plot(kind='bar')
        plt.title(f'Distribution of {col_to_plot}')
        plt.xlabel(col_to_plot)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        img_path = f"barchart_{col_to_plot}.png"
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return f"Created bar chart saved as {img_path}. Top values in {col_to_plot}:\n{value_counts.to_string()}"
    
    def _auto_visualize(self, df: pd.DataFrame) -> str:
        """Automatically create appropriate visualizations"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        visualizations = []
        
        # Create histogram for numeric columns
        if len(numeric_cols) > 0:
            hist_result = self._create_histogram(df, None)
            visualizations.append(hist_result)
        
        # Create correlation heatmap if multiple numeric columns
        if len(numeric_cols) > 1:
            heatmap_result = self._create_heatmap(df)
            visualizations.append(heatmap_result)
        
        # Create bar chart for categorical columns
        if len(categorical_cols) > 0:
            bar_result = self._create_bar_chart(df, None)
            visualizations.append(bar_result)
        
        if visualizations:
            return "Auto-generated visualizations:\n" + "\n".join(visualizations)
        else:
            return "No suitable columns found for visualization"
    
    def _correlation_analysis(self, data_source: str, columns: Optional[List[str]]) -> str:
        """Perform correlation analysis"""
        try:
            df = self._get_dataset(data_source)
            if isinstance(df, str):
                return df
            
            if columns:
                df = df[columns]
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) < 2:
                return "Need at least 2 numeric columns for correlation analysis"
            
            correlation_matrix = df[numeric_cols].corr()
            
            # Find significant correlations
            results = ["CORRELATION ANALYSIS"]
            results.append(f"Correlation Matrix:\n{correlation_matrix.to_string()}")
            
            # Strong correlations
            strong_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_val = correlation_matrix.iloc[i, j]
                    col1, col2 = correlation_matrix.columns[i], correlation_matrix.columns[j]
                    if abs(corr_val) > 0.5:
                        strength = "Strong" if abs(corr_val) > 0.7 else "Moderate"
                        direction = "positive" if corr_val > 0 else "negative"
                        strong_pairs.append(f"{col1} - {col2}: {corr_val:.3f} ({strength} {direction})")
            
            if strong_pairs:
                results.append(f"\nSignificant correlations (|r| > 0.5):\n" + "\n".join(strong_pairs))
            else:
                results.append("\nNo strong correlations found (all |r| < 0.5)")
            
            return "\n".join(results)
            
        except Exception as e:
            return f"Error in correlation analysis: {str(e)}"
    
    def _clustering_analysis(self, data_source: str, columns: Optional[List[str]], 
                           parameters: Optional[Dict[str, Any]]) -> str:
        """Perform K-means clustering analysis"""
        try:
            df = self._get_dataset(data_source)
            if isinstance(df, str):
                return df
            
            if columns:
                df_cluster = df[columns]
            else:
                df_cluster = df.select_dtypes(include=[np.number])
            
            if df_cluster.empty:
                return "No numeric columns available for clustering"
            
            # Handle missing values
            df_cluster = df_cluster.dropna()
            
            # Scale the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df_cluster)
            
            # Get number of clusters
            n_clusters = parameters.get('n_clusters', 3) if parameters else 3
            
            # Perform K-means
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(scaled_data)
            
            # Add cluster labels to original data
            df_result = df_cluster.copy()
            df_result['Cluster'] = clusters
            
            # Analyze clusters
            results = [f"K-MEANS CLUSTERING ANALYSIS (k={n_clusters})"]
            results.append(f"Data shape: {df_cluster.shape}")
            results.append(f"Features used: {list(df_cluster.columns)}")
            
            # Cluster summary
            cluster_summary = df_result.groupby('Cluster').agg(['mean', 'count']).round(3)
            results.append(f"\nCluster Summary:\n{cluster_summary.to_string()}")
            
            # Cluster sizes
            cluster_sizes = pd.Series(clusters).value_counts().sort_index()
            results.append(f"\nCluster Sizes:\n{cluster_sizes.to_string()}")
            
            # Inertia (within-cluster sum of squares)
            results.append(f"\nInertia (WCSS): {kmeans.inertia_:.3f}")
            
            return "\n".join(results)
            
        except Exception as e:
            return f"Error in clustering analysis: {str(e)}"
    
    def _pca_analysis(self, data_source: str, columns: Optional[List[str]], 
                     parameters: Optional[Dict[str, Any]]) -> str:
        """Perform Principal Component Analysis"""
        try:
            df = self._get_dataset(data_source)
            if isinstance(df, str):
                return df
            
            if columns:
                df_pca = df[columns]
            else:
                df_pca = df.select_dtypes(include=[np.number])
            
            if df_pca.empty:
                return "No numeric columns available for PCA"
            
            # Handle missing values
            df_pca = df_pca.dropna()
            
            # Scale the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df_pca)
            
            # Get number of components
            n_components = parameters.get('n_components', min(5, df_pca.shape[1])) if parameters else min(5, df_pca.shape[1])
            
            # Perform PCA
            pca = PCA(n_components=n_components)
            pca_result = pca.fit_transform(scaled_data)
            
            results = [f"PRINCIPAL COMPONENT ANALYSIS"]
            results.append(f"Original features: {list(df_pca.columns)}")
            results.append(f"Number of components: {n_components}")
            
            # Explained variance
            explained_var = pca.explained_variance_ratio_
            cumulative_var = np.cumsum(explained_var)
            
            variance_df = pd.DataFrame({
                'Component': [f'PC{i+1}' for i in range(n_components)],
                'Explained Variance': explained_var,
                'Cumulative Variance': cumulative_var
            })
            results.append(f"\nExplained Variance:\n{variance_df.to_string(index=False)}")
            
            # Component loadings
            loadings = pd.DataFrame(
                pca.components_.T,
                columns=[f'PC{i+1}' for i in range(n_components)],
                index=df_pca.columns
            )
            results.append(f"\nComponent Loadings:\n{loadings.round(3).to_string()}")
            
            return "\n".join(results)
            
        except Exception as e:
            return f"Error in PCA analysis: {str(e)}"
    
    def _statistical_analysis(self, data_source: str, columns: Optional[List[str]], 
                            parameters: Optional[Dict[str, Any]]) -> str:
        """Perform advanced statistical analysis"""
        try:
            df = self._get_dataset(data_source)
            if isinstance(df, str):
                return df
            
            if columns:
                df_stats = df[columns]
            else:
                df_stats = df
            
            numeric_cols = df_stats.select_dtypes(include=[np.number]).columns
            
            results = ["ADVANCED STATISTICAL ANALYSIS"]
            
            # Normality tests
            if len(numeric_cols) > 0:
                results.append("\nNORMALITY TESTS (Shapiro-Wilk):")
                for col in numeric_cols[:5]:  # Test first 5 columns
                    data = df_stats[col].dropna()
                    if len(data) > 3:
                        stat, p_value = stats.shapiro(data)
                        is_normal = p_value > 0.05
                        results.append(f"{col}: statistic={stat:.4f}, p-value={p_value:.4f}, normal={is_normal}")
            
            # T-tests (if applicable)
            if len(numeric_cols) >= 2:
                results.append("\nT-TESTS (comparing first two numeric columns):")
                col1, col2 = numeric_cols[0], numeric_cols[1]
                data1, data2 = df_stats[col1].dropna(), df_stats[col2].dropna()
                
                if len(data1) > 1 and len(data2) > 1:
                    # Independent t-test
                    stat, p_value = stats.ttest_ind(data1, data2)
                    results.append(f"Independent t-test ({col1} vs {col2}): statistic={stat:.4f}, p-value={p_value:.4f}")
                    
                    # Paired t-test (if same length)
                    if len(data1) == len(data2):
                        stat, p_value = stats.ttest_rel(data1, data2)
                        results.append(f"Paired t-test ({col1} vs {col2}): statistic={stat:.4f}, p-value={p_value:.4f}")
            
            # Chi-square test for categorical variables
            categorical_cols = df_stats.select_dtypes(include=['object']).columns
            if len(categorical_cols) >= 2:
                results.append("\nCHI-SQUARE TESTS:")
                col1, col2 = categorical_cols[0], categorical_cols[1]
                contingency_table = pd.crosstab(df_stats[col1], df_stats[col2])
                chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
                results.append(f"Chi-square test ({col1} vs {col2}): chi2={chi2:.4f}, p-value={p_value:.4f}, dof={dof}")
            
            return "\n".join(results)
            
        except Exception as e:
            return f"Error in statistical analysis: {str(e)}"
    
    def _get_dataset(self, data_source: str) -> Union[pd.DataFrame, str]:
        """Get dataset from storage"""
        if data_source in self._data_store:
            return self._data_store[data_source]
        else:
            # Try to load if it's a file path
            try:
                if data_source.endswith('.csv'):
                    df = pd.read_csv(data_source)
                    self._data_store[data_source] = df
                    return df
                elif data_source.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(data_source)
                    self._data_store[data_source] = df
                    return df
            except Exception:
                pass
            
            return f"Dataset not found: {data_source}. Available datasets: {list(self._data_store.keys())}"
