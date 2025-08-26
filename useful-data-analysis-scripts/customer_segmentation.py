# =====================================================
# SCRIPT 3: Customer Segmentation & Profile Generator
# =====================================================

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class CustomerSegmenter:
    def __init__(self, customer_data):
        self.data = customer_data.copy()
        self.segments = None
        self.segment_profiles = None
    
    def rfm_analysis(self, customer_id_col, order_date_col, revenue_col):
        """Perform RFM (Recency, Frequency, Monetary) analysis"""
        # Calculate Recency
        max_date = self.data[order_date_col].max()
        rfm = self.data.groupby(customer_id_col).agg({
            order_date_col: lambda x: (max_date - x.max()).days,
            revenue_col: ['count', 'sum']
        })
        
        rfm.columns = ['Recency', 'Frequency', 'Monetary']
        
        # Create RFM scores
        rfm['R_Score'] = pd.qcut(rfm['Recency'].rank(method='first'), 5, labels=[5,4,3,2,1])
        rfm['F_Score'] = pd.qcut(rfm['Frequency'], 5, labels=[1,2,3,4,5])
        rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1,2,3,4,5])
        
        rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
        
        # Segment customers
        rfm['Segment'] = rfm['RFM_Score'].apply(self._categorize_rfm)
        
        self.rfm_data = rfm
        return rfm
    
    def behavioral_clustering(self, features, n_clusters=5):
        """Perform K-means clustering on behavioral features"""
        # Prepare data
        X = self.data[features].fillna(0)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        self.data['Cluster'] = clusters
        self.segments = clusters
        
        return clusters
    
    def generate_segment_profiles(self, segment_col='Segment'):
        """Generate detailed profiles for each segment"""
        profiles = {}
        
        for segment in self.data[segment_col].unique():
            segment_data = self.data[self.data[segment_col] == segment]
            
            # Calculate segment characteristics
            profile = {
                'size': len(segment_data),
                'percentage': len(segment_data) / len(self.data) * 100,
                'avg_monetary': segment_data.select_dtypes(include=[np.number]).mean().to_dict(),
                'demographics': {}
            }
            
            # Add categorical summaries
            categorical_cols = segment_data.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if col != segment_col:
                    profile['demographics'][col] = segment_data[col].mode().iloc[0] if len(segment_data[col].mode()) > 0 else 'N/A'
            
            # Generate recommendations
            profile['recommendations'] = self._generate_recommendations(segment, profile)
            
            profiles[segment] = profile
        
        self.segment_profiles = profiles
        return profiles
    
    def _categorize_rfm(self, score):
        """Categorize RFM scores into meaningful segments"""
        if score in ['555', '554', '544', '545', '454', '455', '445']:
            return 'Champions'
        elif score in ['543', '444', '435', '355', '354', '345', '344']:
            return 'Loyal Customers'
        elif score in ['553', '551', '552', '541', '542', '533', '532', '531', '452', '451']:
            return 'Potential Loyalists'
        elif score in ['512', '511', '422', '421', '412', '411', '311']:
            return 'New Customers'
        elif score in ['155', '154', '144', '214', '215', '115', '114']:
            return 'At Risk'
        elif score in ['155', '254', '245']:
            return "Can't Lose Them"
        else:
            return 'Others'
    
    def _generate_recommendations(self, segment, profile):
        """Generate actionable recommendations for each segment"""
        recommendations = []
        
        if segment == 'Champions':
            recommendations = [
                'Reward them for their loyalty',
                'Ask for reviews and referrals',
                'Offer exclusive early access to new products'
            ]
        elif segment == 'At Risk':
            recommendations = [
                'Send personalized reactivation campaigns',
                'Offer special discounts to win them back',
                'Conduct surveys to understand their concerns'
            ]
        elif segment == 'New Customers':
            recommendations = [
                'Provide excellent onboarding experience',
                'Educate them about product features',
                'Build relationship through targeted content'
            ]
        
        return recommendations

# Example usage
segmenter = CustomerSegmenter(customer_data)
rfm_results = segmenter.rfm_analysis('customer_id', 'order_date', 'revenue')
profiles = segmenter.generate_segment_profiles()

for segment, profile in profiles.items():
    print(f"\n{segment}: {profile['size']} customers ({profile['percentage']:.1f}%)")
    for rec in profile['recommendations']:
        print(f"  - {rec}")
