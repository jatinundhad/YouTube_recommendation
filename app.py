from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

data = pd.read_csv("preprocessing/preprocessed_data.csv")
categories = data["category_name"].unique()


@app.route("/")
def recommedation():

    thumbnails = data[["thumbnail_link", "title",
                       "channel_title", "category_name"]].head(3).values.tolist()

    return render_template("home.html", thumbnails=thumbnails, categories=categories)


@app.route("/suggestions", methods=["POST"])
def suggestions():
    if request.method == "POST":
        selected_categories = request.form.keys()

        mainDF = pd.DataFrame()
        for category in selected_categories:
            df = data[data["category_name"] == category]
            if df.shape[0] >= 5:
                df = df.sample(n=5)

            mainDF = pd.concat([df, mainDF])

        mainDF = mainDF.sample(frac=1)

        thumbnails = mainDF[["thumbnail_link", "title",
                             "channel_title", "category_name", "views", "likes", "dislikes", "comment_count"]].values.tolist()

        return render_template("Result.html", thumbnails=thumbnails)


@app.route("/recommend", methods=["POST"])
def recommend():
    if request.method == "POST":
        rec_videos = data[data["category_name"]
                          == request.form["category"]]

        refined_data = rec_videos[[
            "views", "likes", "dislikes", "comment_count"]]

        refined_data_array = refined_data.to_numpy()

        query = [int(request.form["views"]), int(request.form["likes"]),
                 int(request.form["dislikes"]), int(request.form["comment_count"])]

        cosine_similarities = []

        for row in refined_data_array:
            cosine_similarities.append(cosine_similarity([query], [row])[0][0])

        columns = rec_videos.columns.to_list()
        columns.append("cosine_similarity")

        rec_videos_array = rec_videos.to_numpy()

        new_arr = np.concatenate(
            (rec_videos_array, np.expand_dims(cosine_similarities, axis=1)), axis=1)

        rec_videos = pd.DataFrame(new_arr, columns=columns)

        final = rec_videos.sort_values(
            by=["cosine_similarity"], ascending=False).head(10)

        print(final)

        return render_template("Recommendation.html")


app.run(debug=True)
