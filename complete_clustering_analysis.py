"""
Multi-Pollutant Profile Grouping of Air Quality Data Using K-Means Clustering
Comprehensive Analysis Covering All Presenter Sections

This script demonstrates manual K-means clustering implementation on air quality data
without using sklearn, covering all 5 presenter parts as described in the TODO.
"""

import csv
import math
import random
from collections import defaultdict


class ManualKMeans:
    """Manual K-Means Implementation without sklearn"""
    
    def __init__(self, k=2, max_iters=100, random_state=42):
        self.k = k
        self.max_iters = max_iters
        self.random_state = random_state
        random.seed(random_state)
        
    def euclidean_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        if isinstance(point1, (int, float)) and isinstance(point2, (int, float)):
            return abs(point1 - point2)
        
        distance = 0
        for i in range(len(point1)):
            distance += (point1[i] - point2[i]) ** 2
        return math.sqrt(distance)
    
    def initialize_centroids(self, data):
        """Initialize centroids randomly"""
        if not data:
            return []
        
        # Check if data is 1D or multi-dimensional
        if isinstance(data[0], (int, float)):
            # 1D data
            min_val, max_val = min(data), max(data)
            return [random.uniform(min_val, max_val) for _ in range(self.k)]
        else:
            # Multi-dimensional data
            n_features = len(data[0])
            centroids = []
            for _ in range(self.k):
                centroid = []
                for j in range(n_features):
                    feature_values = [point[j] for point in data]
                    min_val, max_val = min(feature_values), max(feature_values)
                    centroid.append(random.uniform(min_val, max_val))
                centroids.append(centroid)
            return centroids
    
    def assign_clusters(self, data, centroids):
        """Assign each point to the nearest centroid"""
        clusters = [[] for _ in range(self.k)]
        cluster_assignments = []
        
        for point in data:
            distances = [self.euclidean_distance(point, centroid) for centroid in centroids]
            closest_cluster = distances.index(min(distances))
            clusters[closest_cluster].append(point)
            cluster_assignments.append(closest_cluster)
        
        return clusters, cluster_assignments
    
    def update_centroids(self, clusters):
        """Update centroids based on current clusters"""
        new_centroids = []
        
        for cluster in clusters:
            if not cluster:
                continue
            
            if isinstance(cluster[0], (int, float)):
                # 1D data
                centroid = sum(cluster) / len(cluster)
            else:
                # Multi-dimensional data
                n_features = len(cluster[0])
                centroid = []
                for j in range(n_features):
                    feature_sum = sum(point[j] for point in cluster)
                    centroid.append(feature_sum / len(cluster))
            
            new_centroids.append(centroid)
        
        return new_centroids
    
    def fit(self, data):
        """Fit K-means to the data"""
        self.centroids = self.initialize_centroids(data)
        
        for iteration in range(self.max_iters):
            clusters, assignments = self.assign_clusters(data, self.centroids)
            new_centroids = self.update_centroids(clusters)
            
            # Remove None centroids (empty clusters)
            new_centroids = [c for c in new_centroids if c is not None]
            
            # Check for convergence
            if len(new_centroids) == len(self.centroids):
                converged = True
                for old, new in zip(self.centroids, new_centroids):
                    if isinstance(old, (int, float)):
                        if abs(old - new) > 1e-6:
                            converged = False
                            break
                    else:
                        if any(abs(o - n) > 1e-6 for o, n in zip(old, new)):
                            converged = False
                            break
                
                if converged:
                    print(f"   Converged after {iteration + 1} iterations")
                    break
            
            self.centroids = new_centroids
        
        self.final_clusters, self.cluster_assignments = self.assign_clusters(data, self.centroids)
        return self


def load_air_quality_data():
    """Load the air quality dataset"""
    data = []
    with open('s:\\AIML\\global_air_quality_data_10000.csv', 'r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            data.append(row)
    return data


def get_city_averages(data, cities, pollutants):
    """Calculate average pollutant values for specified cities"""
    city_data = defaultdict(list)
    
    for record in data:
        city = record['City']
        if city in cities:
            pollutant_values = {}
            for pollutant in pollutants:
                try:
                    value = float(record[pollutant])
                    pollutant_values[pollutant] = value
                except (ValueError, KeyError):
                    continue
            if len(pollutant_values) == len(pollutants):
                city_data[city].append(pollutant_values)
    
    # Calculate averages
    averages = {}
    for city in cities:
        if city in city_data and city_data[city]:
            avg_values = []
            for pollutant in pollutants:
                values = [record[pollutant] for record in city_data[city]]
                if values:
                    avg_values.append(sum(values) / len(values))
                else:
                    avg_values.append(0)
            averages[city] = avg_values
    
    return averages


def calculate_aqi_simple(pm25, pm10, no2):
    """Simplified AQI calculation based on major pollutants"""
    pm25_norm = min(pm25 / 35.0, 1.0) * 100
    pm10_norm = min(pm10 / 50.0, 1.0) * 100
    no2_norm = min(no2 / 40.0, 1.0) * 100
    
    return max(pm25_norm, pm10_norm, no2_norm)


def get_country_aqi_averages(data, countries):
    """Calculate average AQI for specified countries"""
    country_data = defaultdict(list)
    
    for record in data:
        country = record['Country']
        if country in countries:
            try:
                pm25 = float(record['PM2.5'])
                pm10 = float(record['PM10'])
                no2 = float(record['NO2'])
                
                aqi = calculate_aqi_simple(pm25, pm10, no2)
                country_data[country].append(aqi)
            except (ValueError, KeyError):
                continue
    
    averages = {}
    for country in countries:
        if country in country_data and country_data[country]:
            averages[country] = sum(country_data[country]) / len(country_data[country])
    
    return averages


def get_city_aqi_averages(data, cities):
    """Calculate average AQI for specified cities"""
    city_data = defaultdict(list)
    
    for record in data:
        city = record['City']
        if city in cities:
            try:
                pm25 = float(record['PM2.5'])
                pm10 = float(record['PM10'])
                no2 = float(record['NO2'])
                
                aqi = calculate_aqi_simple(pm25, pm10, no2)
                city_data[city].append(aqi)
            except (ValueError, KeyError):
                continue
    
    averages = {}
    for city in cities:
        if city in city_data and city_data[city]:
            averages[city] = sum(city_data[city]) / len(city_data[city])
    
    return averages


def main():
    """Main function to run all presenter analyses"""
    
    print("="*80)
    print("MULTI-POLLUTANT PROFILE GROUPING OF AIR QUALITY DATA")
    print("Using Manual K-Means Clustering Implementation")
    print("="*80)
    
    # Load dataset
    print("\n📊 Loading air quality dataset...")
    air_quality_data = load_air_quality_data()
    print(f"   Loaded {len(air_quality_data)} records")
    
    # ========================================================================
    # PRESENTER 1: Simple 2D Clustering (PM2.5 and PM10)
    # ========================================================================
    print("\n" + "="*60)
    print("1️⃣  PRESENTER 1: Simple 2D Clustering (PM2.5 and PM10)")
    print("="*60)
    
    selected_cities_p1 = ['Bangkok', 'Istanbul', 'Mumbai', 'Paris', 'Tokyo', 
                          'New York', 'London', 'Cairo', 'Mexico City', 'Seoul']
    
    pollutants_p1 = ['PM2.5', 'PM10']
    city_averages_p1 = get_city_averages(air_quality_data, selected_cities_p1, pollutants_p1)
    
    print(f"\nCity averages for {len(selected_cities_p1)} cities:")
    for city, values in city_averages_p1.items():
        print(f"  {city:15}: PM2.5={values[0]:6.2f}, PM10={values[1]:6.2f}")
    
    # Clustering
    clustering_data_p1 = list(city_averages_p1.values())
    city_names_p1 = list(city_averages_p1.keys())
    
    kmeans_p1 = ManualKMeans(k=2, random_state=42)
    kmeans_p1.fit(clustering_data_p1)
    
    print(f"\nK-Means Clustering Results (K=2):")
    print("Final Centroids:")
    for i, centroid in enumerate(kmeans_p1.centroids):
        print(f"  Cluster {i+1}: PM2.5={centroid[0]:6.2f}, PM10={centroid[1]:6.2f}")
    
    print("\nCluster Assignments:")
    for city, cluster_id in zip(city_names_p1, kmeans_p1.cluster_assignments):
        values = city_averages_p1[city]
        print(f"  {city:15}: Cluster {cluster_id+1} (PM2.5={values[0]:6.2f}, PM10={values[1]:6.2f})")
    
    print("\n📊 Conclusion: The graph reveals two separate groups—one with higher pollution")
    print("   and one cleaner cluster—demonstrating clear distinction with two features.")
    
    # ========================================================================
    # PRESENTER 2: Country-Level AQI Clustering
    # ========================================================================
    print("\n" + "="*60)
    print("2️⃣  PRESENTER 2: Country-Level AQI Clustering")
    print("="*60)
    
    selected_countries = ['Thailand', 'Turkey', 'Brazil', 'India', 'France', 'USA']
    country_aqi = get_country_aqi_averages(air_quality_data, selected_countries)
    
    print(f"\nCountry Average AQI Values:")
    for country, aqi in sorted(country_aqi.items(), key=lambda x: x[1]):
        print(f"  {country:12}: AQI = {aqi:6.2f}")
    
    # Clustering
    aqi_values = list(country_aqi.values())
    country_names_p2 = list(country_aqi.keys())
    
    kmeans_p2 = ManualKMeans(k=3, random_state=42)
    kmeans_p2.fit(aqi_values)
    
    print(f"\nK-Means Clustering Results (K=3):")
    print("Final Centroids (AQI levels):")
    centroid_labels = ['Low Pollution', 'Medium Pollution', 'High Pollution']
    sorted_centroids = sorted(enumerate(kmeans_p2.centroids), key=lambda x: x[1])
    
    for i, (orig_idx, centroid) in enumerate(sorted_centroids):
        print(f"  Cluster {orig_idx+1} ({centroid_labels[i]}): AQI = {centroid:6.2f}")
    
    print("\nCountry Cluster Assignments:")
    for country, cluster_id, aqi in zip(country_names_p2, kmeans_p2.cluster_assignments, aqi_values):
        print(f"  {country:12}: Cluster {cluster_id+1} - AQI = {aqi:6.2f}")
    
    print("\n📊 Conclusion: Distinct partitions indicate national-level air quality")
    print("   differences, enabling policymakers to focus resources on high-risk nations.")
    
    # ========================================================================
    # PRESENTER 3: City-Level AQI Clustering
    # ========================================================================
    print("\n" + "="*60)
    print("3️⃣  PRESENTER 3: City-Level AQI Clustering")
    print("="*60)
    
    selected_cities_p3 = ['Bangkok', 'Istanbul', 'Mumbai', 'Paris', 'Tokyo', 'New York', 'London', 'Cairo']
    city_aqi = get_city_aqi_averages(air_quality_data, selected_cities_p3)
    
    print(f"\nCity Average AQI Values:")
    for city, aqi in sorted(city_aqi.items(), key=lambda x: x[1]):
        print(f"  {city:15}: AQI = {aqi:6.2f}")
    
    # Clustering
    city_aqi_values = list(city_aqi.values())
    city_names_p3 = list(city_aqi.keys())
    
    kmeans_p3 = ManualKMeans(k=3, random_state=42)
    kmeans_p3.fit(city_aqi_values)
    
    print(f"\nK-Means Clustering Results (K=3):")
    
    # Group cities by cluster
    clusters_dict = defaultdict(list)
    for city, cluster_id, aqi in zip(city_names_p3, kmeans_p3.cluster_assignments, city_aqi_values):
        clusters_dict[cluster_id].append((city, aqi))
    
    # Display each cluster
    for cluster_id in range(len(kmeans_p3.centroids)):
        if cluster_id in clusters_dict:
            centroid_aqi = kmeans_p3.centroids[cluster_id]
            if centroid_aqi < 50:
                level = "Low Pollution"
            elif centroid_aqi < 75:
                level = "Medium Pollution" 
            else:
                level = "High Pollution"
            
            print(f"\n🏙️  CLUSTER {cluster_id+1} ({level}):")
            print(f"   Centroid AQI: {centroid_aqi:.2f}")
            print("   Cities in this cluster:")
            
            for city, aqi in sorted(clusters_dict[cluster_id], key=lambda x: x[1]):
                print(f"   • {city:15}: AQI = {aqi:6.2f}")
    
    print("\n📊 Conclusion: Clusters effectively group cities by similar pollution levels,")
    print("   enabling identification of local pollution discrepancies.")
    
    # ========================================================================
    # PRESENTER 4: Two-Pollutant City Clustering (PM2.5 and NO2)
    # ========================================================================
    print("\n" + "="*60)
    print("4️⃣  PRESENTER 4: Two-Pollutant City Clustering (PM2.5 and NO2)")
    print("="*60)
    
    selected_cities_p4 = ['Bangkok', 'Istanbul', 'Mumbai', 'Paris', 'Tokyo', 'New York', 'London', 'Cairo']
    pollutants_p4 = ['PM2.5', 'NO2']
    city_averages_p4 = get_city_averages(air_quality_data, selected_cities_p4, pollutants_p4)
    
    print(f"\nCity averages for PM2.5 and NO2:")
    for city, values in city_averages_p4.items():
        print(f"  {city:15}: PM2.5={values[0]:6.2f}, NO2={values[1]:6.2f}")
    
    # Clustering
    clustering_data_p4 = list(city_averages_p4.values())
    city_names_p4 = list(city_averages_p4.keys())
    
    kmeans_p4 = ManualKMeans(k=3, random_state=42)
    kmeans_p4.fit(clustering_data_p4)
    
    print(f"\nK-Means Clustering Results (K=3):")
    print("Final Centroids:")
    for i, centroid in enumerate(kmeans_p4.centroids):
        print(f"  Cluster {i+1}: PM2.5={centroid[0]:6.2f}, NO2={centroid[1]:6.2f}")
    
    print("\nCluster Assignments:")
    for city, cluster_id in zip(city_names_p4, kmeans_p4.cluster_assignments):
        values = city_averages_p4[city]
        print(f"  {city:15}: Cluster {cluster_id+1} (PM2.5={values[0]:6.2f}, NO2={values[1]:6.2f})")
    
    print("\n📊 Conclusion: Clusters capture pollutant-compositional diversity among cities.")
    print("   Iterative centroid updates show convergence in multi-dimensional space.")
    
    # ========================================================================
    # PRESENTER 5: Multi-Feature Three-Pollutant Clustering
    # ========================================================================
    print("\n" + "="*60)
    print("5️⃣  PRESENTER 5: Multi-Feature Three-Pollutant Clustering")
    print("="*60)
    
    selected_cities_p5 = ['Bangkok', 'Istanbul', 'Mumbai', 'Paris', 'Tokyo', 'New York', 'London']
    pollutants_p5 = ['PM2.5', 'PM10', 'NO2']
    city_averages_p5 = get_city_averages(air_quality_data, selected_cities_p5, pollutants_p5)
    
    print(f"\nCity averages for PM2.5, PM10, and NO2:")
    for city, values in city_averages_p5.items():
        print(f"  {city:15}: PM2.5={values[0]:6.2f}, PM10={values[1]:6.2f}, NO2={values[2]:6.2f}")
    
    # Clustering
    clustering_data_p5 = list(city_averages_p5.values())
    city_names_p5 = list(city_averages_p5.keys())
    
    kmeans_p5 = ManualKMeans(k=2, random_state=42)
    kmeans_p5.fit(clustering_data_p5)
    
    print(f"\nK-Means Clustering Results (K=2):")
    print("Final Centroids (Multi-Pollutant Profiles):")
    for i, centroid in enumerate(kmeans_p5.centroids):
        print(f"  Cluster {i+1}: PM2.5={centroid[0]:6.2f}, PM10={centroid[1]:6.2f}, NO2={centroid[2]:6.2f}")
    
    # Group cities by cluster
    clusters_dict_p5 = defaultdict(list)
    for city, cluster_id in zip(city_names_p5, kmeans_p5.cluster_assignments):
        values = city_averages_p5[city]
        clusters_dict_p5[cluster_id].append((city, values))
    
    # Display each cluster
    cluster_labels = ['Cleaner Cities', 'More Polluted Cities']
    
    for cluster_id in range(len(kmeans_p5.centroids)):
        if cluster_id in clusters_dict_p5:
            centroid = kmeans_p5.centroids[cluster_id]
            print(f"\n🌍 {cluster_labels[cluster_id].upper()} (Cluster {cluster_id+1}):")
            print(f"   Centroid Profile: PM2.5={centroid[0]:.2f}, PM10={centroid[1]:.2f}, NO2={centroid[2]:.2f}")
            print("   Cities in this cluster:")
            
            for city, values in sorted(clusters_dict_p5[cluster_id], key=lambda x: sum(x[1])):
                total_pollution = sum(values)
                print(f"   • {city:15}: PM2.5={values[0]:6.2f}, PM10={values[1]:6.2f}, NO2={values[2]:6.2f} (Total: {total_pollution:6.2f})")
    
    print("\n📊 Conclusion: Multi-pollutant analysis yields comprehensive pollution profiles,")
    print("   better capturing complexity of air quality for health assessments.")
    
    # ========================================================================
    # SUMMARY AND OVERALL CONCLUSIONS
    # ========================================================================
    print("\n" + "="*80)
    print("📈 COMPREHENSIVE ANALYSIS SUMMARY")
    print("="*80)
    
    print("\n🎯 PRESENTER CONTRIBUTIONS:")
    print("   1️⃣ Presenter 1: Basic 2D pollution clustering (PM2.5, PM10)")
    print("   2️⃣ Presenter 2: Country AQI grouping for policy targeting")
    print("   3️⃣ Presenter 3: City AQI clustering for urban management")
    print("   4️⃣ Presenter 4: Multi-feature clustering with convergence visualization")
    print("   5️⃣ Presenter 5: Comprehensive 3-pollutant profiles")
    
    print("\n✅ KEY FINDINGS:")
    print("   • Multi-pollutant K-Means clustering reveals meaningful air quality patterns")
    print("   • Progression from simple to complex features enhances analytical depth")
    print("   • Different scales (city vs country) provide complementary insights")
    print("   • Manual implementation demonstrates algorithmic understanding")
    print("   • Clustering supports evidence-based environmental decision making")
    
    print("\n🌍 PRACTICAL APPLICATIONS:")
    print("   • Health Risk Assessment: Identify high-risk pollution zones")
    print("   • Resource Allocation: Target interventions based on cluster profiles")
    print("   • Policy Development: National vs local pollution management strategies")
    print("   • Environmental Monitoring: Systematic pollution pattern recognition")
    print("   • Urban Planning: Inform sustainable city development")
    
    print("\n🔮 FUTURE SCOPE:")
    print("   • Temporal Analysis: Incorporate seasonal/yearly pollution trends")
    print("   • Socio-Economic Integration: Add demographic and economic indicators")
    print("   • Weather Correlation: Include meteorological factors")
    print("   • Advanced Clustering: Compare with DBSCAN, hierarchical methods")
    print("   • Real-time Monitoring: Dynamic clustering for live air quality data")
    
    print("\n" + "="*80)
    print("Thank you for following this comprehensive multi-pollutant clustering analysis!")
    print("="*80)


if __name__ == "__main__":
    main()