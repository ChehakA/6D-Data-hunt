Python 3.12.3 (v3.12.3:f6650f9ad7, Apr  9 2024, 08:18:47) [Clang 13.0.0 (clang-1300.0.29.30)] on darwin
Type "help", "copyright", "credits" or "license()" for more information.
╭────────
____ __ 
/ __/__ ___ _____/ /__ 
_\ \/ _ \/ _ `/ __/ '_/ 
/__ / .__/\_,_/_/ /_/\_\ version 3.5.1 
/_/ 
Using Python version 3.9.13 (main, Aug 25 2022 23:26:10) 
Spark context Web UI available at http://r001.ib.bridges2.psc.edu:4041 Spark context available as 'sc' (master = local[*], app id = local-1731280261603). SparkSession available as 'spark'. 
data = sc.textFile("space.dat") #reads in the data 
data.map(lambda line: [float(x.strip(',')) for x in line.split()]) # splits line to element, removes commas, converts string to a floating point number 
Out[2]: PythonRDD[2] at RDD at PythonRDD.scala:53 
df = data.map(lambda line: list(map(lambda x: float(x.replace(",", "")), line.split()))).toDF(["x1", "x2", "x3", "x4", "x5", 
...: ...: "x6"]) #transforms RDD to a data frame 
...: 
df.show() #display the first 20 rows of data frame 
df_location = df.select("x1", "x2", "x3", "x4", "x5", "x6") 
df_location.show() 
from pyspark.ml.feature import StandardScaler #standardize features, mean as 0 and SD as 1 from pyspark.ml.linalg import Vectors #importing to create vectors. 
from pyspark.sql.functions import col #to perform transformation 
from pyspark.ml.feature import VectorAssembler #single vector formation as input for rows
vector_col = VectorAssembler(inputCols = ["x1", "x2", "x3", "x4", "x5", "x6"], outputCol = "features") #single vector column created to input data, inpu tcols is a list of columns with raw feature values, outputcol is the name of new column 
df_features = vector_col.transform(df) #combines cols into a new vector column df_features.show() 
from pyspark.ml.clustering import KMeans 
kmeans = KMeans(k=6, featuresCol="features", predictionCol="cluster") kmeans_model = kmeans.fit(df_features) 
df_clusters = kmeans_model.transform(df_features) 
clustered_data = df_clusters.select("x1", "x2", "x3", "x4", "x5", "x6", "cluster").co ...: llect() 
for row in clustered_data[:50]: 
...: print(f"Location: {row[:6]}, Cluster: {row['cluster']}") 
centroids = kmeans_model.clusterCenters() 
for i, centroid in enumerate(centroids): 
...: print(f"Cluster {i} Centroid: {centroid}") 
Cluster 0 Centroid: [14.99656549 80.01182547 14.99467628 80.05577159 14.97239517 80.02087051] 
Cluster 1 Centroid: [0.08449107 0.08178346 0.08159672 0.07987057 0.0867673 0.08799652] Cluster 2 Centroid: [70.00305127 60.02333638 50.02662094 39.99197196 29.99459599 20.01922986] 
Cluster 3 Centroid: [25.00130508 25.00506354 24.97928291 75.01356403 75.01196977 75.00804088] 
Cluster 4 Centroid: [74.97159526 75.00501487 74.95916543 75.00238108 74.98414552 74.99371641] 
Cluster 5 Centroid: [14.87519246 14.87569221 14.87619196 14.87669171 14.87719146 14.87769121] 
scaler = StandardScaler(inputCol = "features", outputCol = "scaled_features", withMean = True, withStd= True) #scaling 
scaler_model = scaler.fit(df_features) #standardizing the data
df_scaled = scaler_model.transform(df_features) #transformed dataframe where mean is 0 and standard deviation of 1. 
df_scaled.show() #displays first 20 rows 
rom pyspark.ml.feature import PCA 
File "<ipython-input-15-3a7209190178>", line 1 
rom pyspark.ml.feature import PCA 
^ 
SyntaxError: invalid syntax 
from pyspark.ml.feature import PCA # for dimension reduction 
pca = PCA(k=3, inputCol = "scaled_features", outputCol = "pca_features") #reduce the data to 3 components. 
pca_model = pca.fit(df_scaled) #scaled features in the pca_model 
24/11/10 18:19:09 WARN InstanceBuilder: Failed to load implementation from:dev.ludovic.netlib.blas.JNIBLAS 
24/11/10 18:19:10 WARN InstanceBuilder: Failed to load implementation from:dev.ludovic.netlib.lapack.JNILAPACK 
df_pca = pca_model.transform(df_scaled) #pca transformation 
df_pca.show(truncate =False) #show full vectors of reduced dimensions. 
from pyspark.ml.clustering import KMeans #kmeans clustering 
kmeans = KMeans(k = 5, featuresCol = "pca_features", predictionCol = "cluster") #5 clusters kmeans_model = kmeans.fit(df_pca) #clustering 
df_clusters = kmeans_model.transform(df_pca) #PCA reduced features and predicted clusters 
df_clusters.select("pca_features", "cluster").show(truncate = False) #displays the clusters of data frame and the pca features 
+-------------------------------------------------------------+-------+ 
|pca_features |cluster| 
+-------------------------------------------------------------+-------+ 
|[-0.48507647883359745,-1.741125130594291,0.8892416532844538] |0 |
|[-0.35230134888224207,-1.749610331197141,1.0709576720576948] |0 | |[-0.4520963344588759,-1.8069047732665149,1.1366852581583147] |0 | |[2.5494349055217214,0.5028304162196005,-0.11774164118421077] |1 | |[-2.8621190890359083,1.0494324573206442,-0.06121443627197974]|4 | 
|[-0.6047478658414918,-0.6685324930443652,-1.5578575178837735]|3 | |[3.48215323711399,0.4114526667396421,-0.11565038760145113] |1 | |[-0.3441769907801138,1.673811145384068,0.8443629191989104] |2 | |[-0.6024127681615679,-0.679537234911396,-1.5046939995303061] |3 | |[-2.767707351978514,0.9564944605679877,-0.05242106930684226] |4 | |[-0.5841648185772266,-0.6490502513566618,-1.5172968740241406]|3 | |[-0.5240992669964325,-1.7302431099248459,1.0469721221943469] |0 | |[-0.5330244920335498,-1.6381359205138346,0.9999990098969324] |0 | |[-0.3214474287121455,-1.6206945224321108,0.820981268180607] |0 | |[2.3968403862314,0.5171340851774484,-0.11583573606516562] |1 | |[-0.20772891354398187,-1.8197113387164712,0.80372827119195] |0 | |[-0.6122447485685198,-0.6370721876208902,-1.4935863874552973]|3 | |[-2.812178587757645,1.0010692295565677,-0.056297833053988805]|4 | |[2.5530597733811318,0.5024906339645226,-0.11778691577363395] |1 | |[-2.9105813687763913,1.1003485897514669,-0.06466078643023274]|4 | +-------------------------------------------------------------+-------+ 
only showing top 20 rows 
pca_data = df_clusters.select("pca_features", "cluster").collect() #tuple creation with PCA features and cluster 
pca_model = pca.fit(df_scaled) #fit 
df_pca = pca_model.tranform(df_scaled) 
--------------------------------------------------------------------------- 
AttributeError Traceback (most recent call last) 
<ipython-input-30-48675ae25c95> in <module> 
----> 1 df_pca = pca_model.tranform(df_scaled) 
AttributeError: 'PCAModel' object has no attribute 'tranform' 
df_pca = pca_model.transform(df_scaled) #transformation 
x_vals = [row["pca_features"][0] for row in pca_data] #x coordinate 
y_vals = [row["pca_features"][1] for row in pca_data] #y coordinate 
z_vals = [row["pca_features"][2] for row in pca_data] # x coordinate
cluster_label = [row["cluster"] for row in pca_data] #list with cluster labels in pca_data 
figure = plt.figure(figsize=(10,8)) #creates the scatter plot canvas 
--------------------------------------------------------------------------- 
NameError Traceback (most recent call last) 
<ipython-input-36-764883625f8d> in <module> 
----> 1 figure = plt.figure(figsize=(10,8)) 
NameError: name 'plt' is not defined 
import matplotlib.pyplot as plt 
figure = plt.figure(figsize=(10,8)) #creates the scatter plot canvas 
from mpl_toolkits.mplot3d import Axes3D #found about this through chatgpt, I gave chatgpt the prompt “how do i plot something in spark on 3d after pca” 
3d_plot = ax.scatter(x_vals, y_vals, z_vals, c = cluster_labels, cmap = 'virdis', s = 50) File "<ipython-input-40-22ea777a54f5>", line 1 
3d_plot = ax.scatter(x_vals, y_vals, z_vals, c = cluster_labels, cmap = 'virdis', s = 50) ^ 
SyntaxError: invalid syntax 
threed_plot = ax.scatter(x_vals, y_vals, z_vals, c = cluster_labels, cmap = 'virdis', s = 50) --------------------------------------------------------------------------- 
NameError Traceback (most recent call last) 
<ipython-input-41-2db0b4f7524f> in <module> 
----> 1 threed_plot = ax.scatter(x_vals, y_vals, z_vals, c = cluster_labels, cmap = 'virdis', s = 50) NameError: name 'ax' is not defined 
axis = figure.add_subplot(111, projection = '3d') #axis needed to use scatter 
threed_plot = axis.scatter(x_vals, y_vals, z_vals, c = cluster_labels, 
...: cmap = 'viridis', s = 50) 
plt.colorbar(threed_plot, label = 'Cluster') #easier to interpret with colors of the points Out[49]: <matplotlib.colorbar.Colorbar at 0x14f6c84051f0> 
axis.set_xlabel('Pca comp 1') 
Out[50]: Text(0.5, 0, 'Pca comp 1')
axis.set_ylabel('Pca comp 2') 
Out[51]: Text(0.5, 0.5, 'Pca comp 2') 
axis.set_zlabel('Pca comp 3') 
Out[52]: Text(0.5, 0, 'Pca comp 3') 
plt.title('3D projection') 
Out[53]: Text(0.5, 0.92, '3D projection') 
plt.show() 
plt.savefig("threedprojection.png") 
plt.clf() 
centroids = kmeans_model.clusterCenters() #to get the position of clusters for centroid in centroids: 
...: print(centroid) #coordinates 
...: 
[-0.3669967 -1.7277161 0.98392362] 
[ 2.9188783 0.46817161 -0.12244602] 
[-0.32801356 1.67390292 0.85750315] 
[-0.61484829 -0.67017521 -1.53941027] 
cluster_sizes = df_clusters.groupBy("cluster").count().collect() #Get the size of each cluster by counting the number of points in each cluster 
for row in cluster_sizes: #Print the size of each cluster #the coordinates of each centroid in the 6D feature space after normalization 
...: print(f"Cluster {row['cluster']} size: {row['count']}") 
...: 
Cluster 1 size: 4401 
Cluster 3 size: 3501 
Cluster 4 size: 3001 
Cluster 2 size: 2501 
Cluster 0 size: 4000 
In [57]: centroids = kmeans_model.clusterCenters() 
In [58]: for i, centroid in enumerate(centroids): 
...: print(f"Centroid {i} (6D coordinates): {centroid}") 
...: 
Centroid 0 (6D coordinates): [-0.3669967 -1.7277161 0.98392362] 
Centroid 1 (6D coordinates): [ 2.9188783 0.46817161 -0.12244602] 
Centroid 2 (6D coordinates): [-0.32801356 1.67390292 0.85750315] 
Centroid 3 (6D coordinates): [-0.61484829 -0.67017521 -1.53941027]
Centroid 4 (6D coordinates): [-2.80075002 1.00309676 -0.05063298] 
df_points_in_clusters = df_clusters.select("x1", "x2", "x3", "x4", "x5", "x6", "cluster") df_points_in_clusters.show(truncate=False) 
+------------------+------------------+------------------+------------------+------------------+------------------+------- + 
|x1 |x2 |x3 |x4 |x5 |x6 |cluster| +------------------+------------------+------------------+------------------+------------------+------------------+------- + 
|16.175290018360865|79.26158903022598 |15.279677957430495|84.44449050934236 |18.389312435117514|79.87125481957949 |0 | 
|12.441585418179132|81.38351233369806 |17.745524605746372|77.67266089686214 |11.879433413248021|82.51749363148706 |0 | 
|12.612232031217749|85.35689559849533 |17.217141384399465|81.99797307493071 |12.917574288336507|80.80276566517777 |0 | 
|11.213021770627718|11.213021770627718|11.213021770627718|11.213021770627718|11.21 3021770627718|11.213021770627718|1 | 
|76.15301332156801 |75.0499428915535 |76.49786909309103 |75.10331927879925 |75.79480494739 |75.49118728415654 |4 | 
|25.030992305661588|23.32588783792623 |25.62437382971083 |74.9474332286438 |74.05706902435973 |76.10909554713254 |3 | 
|0.0 |0.6182371406185106|0.0 |0.0 |0.0 |0.0 |1 | |69.85013897336076 |60.376026515389874|50.04560657642975 |40.5142379959055 |30.926195002542762|19.50504076758496 |2 | 
|25.49870422329824 |25.436874174304045|24.412825766180248|74.33420723845597 |73.91554429759485 |75.72750934356775 |3 | 
|74.67235806596484 |75.44075076271328 |72.88886010268914 |74.45081265293294 |75.18708265086273 |75.47187355624806 |4 | 
|25.162444935565276|23.515222831359925|26.179871130503116|73.58311171395387 |72.58295001928295 |76.49809947423712 |3 | 
|17.40806587068996 |82.44337438690698 |16.33739058119697 |85.39047758023496 |15.089048389539467|79.26598455959412 |0 | 
|18.51242818673245 |80.434876732633 |18.374789092014584|83.35293760142048 |15.619534396556798|79.87945165951707 |0 | 
|17.426460864708954|74.7300351649736 |14.51398574990737 |76.94748530306433 |17.478145262579783|80.79789218139493 |0 | 
|13.031879892212178|13.031879892212178|13.031879892212178|13.031879892212178|13.0 31879892212178|13.031879892212178|1 | 
|8.721592082978272 |75.9920627502738 |13.138875602369634|80.87383204658387 |17.706146720589324|77.07318384438084 |0 | 
|24.68500972203311 |27.041787831160235|25.41241644075779 |73.9201271433113 |75.31479039743438 |73.4015810386591 |3 | 
|75.35795662277442 |75.23986536407119 |74.63379192132605 |74.77571635004942 |75.45039386417471 |75.45403460937742 |4 |
|11.169814974103035|11.169814974103035|11.169814974103035|11.169814974103035|11.16 9814974103035|11.169814974103035|1 | 
|76.86538841233997 |74.78172332146798 |78.53134743260597 |75.5088852042855 |76.0144043683936 |75.3926982075889 |4 | 
kmeans = KMeans(k=5, featuresCol="features", predictionCol="cluster") #KMeans clustering with the scaled features 
kmeans_model = kmeans.fit(df_scaled) 
centroids = kmeans_model.clusterCenters() 
for i, centroid in enumerate(centroids): 
...: print(f"Centroid {i} (6D coordinates):") 
...: print(f"x1: {centroid[0]}, x2: {centroid[1]}, x3: {centroid[2]}, x4: {centroid[3]}, x5: {centroid[4]}, x6: {centroid[5]}") 
...: 
Centroid 0 (6D coordinates): 
x1: 14.99656548968305, x2: 80.01182546563689, x3: 14.994676278626875, x4: 80.05577159321754, x5: 14.972395172165648, x6: 80.02087051190908 Centroid 1 (6D coordinates): 
x1: 6.809370298975442, x2: 6.808120978038054, x3: 6.808246362289327, x4: 6.807532262021241, x5: 6.81152048270138, x6: 6.812418030553124 
Centroid 2 (6D coordinates): 
x1: 70.00305127366748, x2: 60.02333638248443, x3: 50.026620941293224, x4: 39.99197196212158, x5: 29.994595992911133, x6: 20.01922985734521 Centroid 3 (6D coordinates): 
x1: 25.001305075721312, x2: 25.005063541763338, x3: 24.97928291360096, x4: 75.013564027407, x5: 75.01196977340423, x6: 75.0080408800156 
Centroid 4 (6D coordinates): 
x1: 74.97159526326213, x2: 75.00501486905792, x3: 74.95916542649921, x4: 75.00238107585585, x5: 74.98414551981284, x6: 74.99371641085219 
cluster_counts = df_clusters.groupBy("cluster").count() #later realised that its doing the same thing as size() 
cluster_counts.show() 
+-------+-----+ 
|cluster|count| 
+-------+-----+ 
| 1| 4401| 
| 3| 3501| 
| 4| 3001|
| 2| 2501| 
| 0| 4000| 
df_pandas = df_clusters.toPandas() #Convert the clustered DataFrame into a Pandas DataFrame for further processing 
x_vals = df_pandas["pca_features"].apply(lambda x: x[0]).tolist() 
y_vals = df_pandas["pca_features"].apply(lambda x: x[1]).tolist() 
z_vals = df_pandas["pca_features"].apply(lambda x: x[2]).tolist() 
cluster_labels = df_pandas["cluster"].tolist() # clustered DataFrame into a Pandas DataFrame for further processing 
clustered_data = {i: ([], [], []) for i in range(6)} # dictionary to store the data points for each cluster in 3D space 
for i, label in enumerate(cluster_labels): #Create a dictionary to store the data points for each cluster in 3D space 
...: clustered_data[label][0].append(x_vals[i]) 
...: clustered_data[label][1].append(y_vals[i]) 
...: clustered_data[label][2].append(z_vals[i]) 
...: 
for cluster, (x, y, z) in clustered_data.items(): 
...: fig = plt.figure(figsize=(10, 8)) 
...: ax = fig.add_subplot(111, projection='3d') 
...: ax = fig.add_subplot(111, projection='3d') 
...: ax.scatter(x, y, z, label=f"Cluster {cluster}", s=50) 
...: ax.set_xlabel('Principal Component 1') 
...: ax.set_ylabel('Principal Component 2') 
...: ax.set_zlabel('Principal Component 3') 
...: ax.set_title(f"Cluster {cluster} in 3D PCA Space") 
...: ax.legend() 
...: plt.savefig(f"cluster_{cluster}_3D.png") 
...: plt.show() 
...: plt.clf() 
pwd 
Out[117]: '/jet/home/carora' 
Cluster 0 Centroid: [14.99656549 80.01182547 14.99467628 80.05577159 14.97239517 80.02087051]
Cluster 1 Centroid: [0.08449107 0.08178346 0.08159672 0.07987057 0.0867673 0.08799652] Cluster 2 Centroid: [70.00305127 60.02333638 50.02662094 39.99197196 29.99459599 20.01922986] 
Cluster 3 Centroid: [25.00130508 25.00506354 24.97928291 75.01356403 75.01196977 75.00804088] 
Cluster 4 Centroid: [74.97159526 75.00501487 74.95916543 75.00238108 74.98414552 74.99371641] 
Cluster 5 Centroid: [14.87519246 14.87569221 14.87619196 14.87669171 14.87719146 14.87769121]


Color 
Points 
location 
shape
Cluster 0 
Dark purple 
4000 
[14.99656549 
80.01182547 
14.99467628 
80.05577159 
14.97239517 
80.02087051]
sphere
Cluster 1 
Yellow 
4401 
[0.08449107 
0.08178346 
0.08159672 
0.07987057 
0.0867673 
0.08799652]
star
... Cluster 2 
... Green 
... 2501 
... [70.00305127 
... 60.02333638 
... 50.02662094 
... 39.99197196 
... 29.99459599 
... 20.01922986]
... rectangle
... Cluster 3 
... Blue 
... 3501 
... [74.97159526 
... 75.00501487 
... 74.95916543 
... 75.00238108 
... 74.98414552 
... 74.99371641]
... ellipse
... 
... 
... 
... Cluster4
... Teal green
... 3001
... [14.87519246 
... 14.87569221 
... 14.87619196 
... 14.87669171 
... 14.87719146 
... 14.87769121]
... slab
... 
... 
... 
... 
... 
... 
