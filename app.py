from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly
import json
from model.ASM import generate_association_rules
import plotly.graph_objects as go

app = Flask(__name__)

raw_data = pd.read_csv("preprocessing/INvideos.csv")
data = pd.read_csv("preprocessing/preprocessed_data.csv")
categories = data["category_name"].unique()


@app.route("/")
def recommedation():
    return render_template("home.html", categories=categories)


@app.route("/preprocessing", methods=["GET"])
def preprocessing():
    if request.method == "GET":

        raw_data_columns = raw_data.columns.tolist()
        raw_data_columns = raw_data_columns[0:len(raw_data_columns)-1]
        preprocessed_data_columns = data.columns.tolist()

        raw_data_10 = raw_data.sample(n=5).values.tolist()
        data_10 = data.sample(n=5).values.tolist()

        fig1 = px.pie(data, names='category_name')
        fig1 = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)

        no_videos_categoryWise = data.groupby("category_name", as_index=False).size()
        fig2 = px.bar(no_videos_categoryWise, x='category_name', y='size')
        fig2 = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)

        Entertainment = data[data["category_name"] == "Entertainment"].sort_values(["views", "likes", "comment_count"], ascending=False)

        channels = Entertainment.groupby("channel_title", as_index=False).size()
        channels_top_10 = channels.sort_values(by="size",ascending=False).head(10)

        fig3 = px.bar(channels_top_10, x='channel_title', y='size')
        fig3 = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)

        return render_template("preprocessing.html", fig1=fig1, fig2 = fig2, fig3 = fig3, raw_data_columns=raw_data_columns, raw_data=raw_data_10, preprocessed_data_columns=preprocessed_data_columns, preprocessed_data=data_10)


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

        thumbnails = final.values.tolist()

        columns = ["Title", "Channel Title", "Views", "Likes", "Dislikes",
                   "Comment counts", "Category name", "Cosine Similarity"]

        c1, c2, c3, df1, df2, df3, df4, df5, time1, time2 = generate_association_rules(thumbnails)

        return render_template("Recommendation.html", thumbnails=thumbnails, columns=columns, c1=c1, c2=c2, c3=c3, df1=df1, df2=df2, df3=df3, df4=df4, df5=df5, time1=time1, time2=time2)

if __name__ == '__main__':
    app.run(debug=True)