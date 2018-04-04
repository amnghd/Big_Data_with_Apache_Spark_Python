import findspark
findspark.init()

import sys
from pyspark import SparkConf, SparkContext
from math import sqrt

conf = SparkConf().setMaster("local[*]").setAppName("MovieSimilarities")
sc = SparkContext(conf = conf)

def loadMovieNames():
    movieNames = {}
    with open("c:/SparkCourse/ml-100k/u.ITEM", encoding='ascii', errors='ignore') as f:
        for line in f:
            fields = line.split('|')
            movieNames[int(fields[0])] = fields[1]
    return movieNames

#Python 3 doesn't let you pass around unpacked tuples,
#so we explicitly extract the ratings now.
def makePairs( userRatings ):
    ratings = userRatings[1]
    (movie1, rating1) = ratings[0]
    (movie2, rating2) = ratings[1]
    return ((movie1, movie2), (rating1, rating2))

def filterDuplicates( userRatings ): # very god for removing duplicates in spark
    ratings = userRatings[1]
    (movie1, rating1) = ratings[0]
    (movie2, rating2) = ratings[1]
    return movie1 < movie2 # from two direction of join, keep the one that is alphabetically ahead
#also filters movies with the same name

def filterLowRatings(movieinfo):
    rating = float(movieinfo[1][1])
    return rating>=2  

def computeCosineSimilarity(ratingPairs):
    numPairs = 0
    sum_xx = sum_yy = sum_xy = 0
    for ratingX, ratingY in ratingPairs:
        sum_xx += ratingX * ratingX
        sum_yy += ratingY * ratingY
        sum_xy += ratingX * ratingY
        numPairs += 1

    numerator = sum_xy
    denominator = sqrt(sum_xx) * sqrt(sum_yy)

    score = 0
    if (denominator):
        score = (numerator / (float(denominator)))

    return (score, numPairs)



def computePearsonSimilarity(ratingPairs):
    numPairs = 0
    sum_xx = sum_yy = sum_xy = sum_x = sum_y =  0
    for ratingX, ratingY in ratingPairs:
        sum_xx += ratingX * ratingX
        sum_yy += ratingY * ratingY
        sum_xy += ratingX * ratingY
        sum_x += ratingX
        sum_y += ratingY
        numPairs += 1

    numerator = sum_xy - sum_x * sum_y / numPairs
    denominator = sqrt((sum_xx - (sum_x)**2/numPairs) * ((sum_yy - (sum_y)**2/numPairs)))

    score = 0
    if (denominator):
        score = (numerator / (float(denominator)))

    return (score, numPairs)


print("\nLoading movie names...")
nameDict = loadMovieNames()

data = sc.textFile("file:///SparkCourse/ml-100k/u.data")

# Map ratings to key / value pairs: user ID => movie ID, rating
ratings = data.map(lambda l: l.split()).map(lambda l: (int(l[0]), (int(l[1]), float(l[2])))).filter(filterLowRatings)

# Emit every movie rated together by the same user.
# Self-join to find every combination.
joinedRatings = ratings.join(ratings)
print("\nDataset is self-joined...")
# At this point our RDD consists of userID => ((movieID, rating), (movieID, rating))

# Filter out duplicate pairs
uniqueJoinedRatings = joinedRatings.filter(filterDuplicates)
print("\nDuplicates are filtered...")

# Now key by (movie1, movie2) pairs.
moviePairs = uniqueJoinedRatings.map(makePairs)
print("\nMovie data is anonymized, no userid...")

# We now have (movie1, movie2) => (rating1, rating2)
# Now collect all ratings for each movie pair and compute similarity
moviePairRatings = moviePairs.groupByKey()

# We now have (movie1, movie2) = > (rating1, rating2), (rating1, rating2) ...
# Can now compute similarities.
moviePairSimilarities = moviePairRatings.mapValues(computePearsonSimilarity).cache()

# Save the results if desired
#moviePairSimilarities.sortByKey()
#moviePairSimilarities.saveAsTextFile("movie-sims")




# Extract similarities for the movie we care about that are "good".
scoreThreshold = 0.38
coOccurenceThreshold = 50

movieID = 483 # Casablanca

# Filter for movies with this sim that are "good" as defined by
# our quality thresholds above (filtering all at the same time)
filteredResults = moviePairSimilarities.filter(lambda pairSim: \
    (pairSim[0][0] == movieID or pairSim[0][1] == movieID) \
    and pairSim[1][0] > scoreThreshold and pairSim[1][1] > coOccurenceThreshold)

# Sort by quality score.
results = filteredResults.map(lambda pairSim: (pairSim[1], pairSim[0])).sortByKey(ascending = False).take(10)

print("Top 10 similar movies for " + nameDict[movieID])
for result in results:
    (sim, pair) = result
    # Display the similarity result that isn't the movie we're looking at
    similarMovieID = pair[0]
    if (similarMovieID == movieID):
        similarMovieID = pair[1]
    print(nameDict[similarMovieID] + "\tscore: " + str(sim[0]) + "\tstrength: " + str(sim[1]))