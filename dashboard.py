import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import joblib
import os
import pickle
import requests
import json
import pandas as pd

# read models
model = pickle.load(open("classifier.p", "rb"))
vectorizer = pickle.load(open("vectorizer.p", "rb"))


app = dash.Dash(__name__)
server = app.server
app.config.suppress_callback_exceptions = True


app.layout = html.Div(
    children=[
        html.H1(
            className='bg-grey ten columns offset-by-one columns',
            children=[
                'AI Classification of Unseen News Headlines'
            ]
        ),
        html.Div(
            className='row',
            children=[
                html.Div(
                    className='pretty_container ten columns offset-by-one columns',
                    children=[
                        html.H2('The algorithm was trained on a data set consisting of Economics & Business,'
                                ' Science & Technology, Health & Health Care and Entertainment news articles from the'
                                ' United States. The AI aims to classify unseen news headlines according to these'
                                ' four clusters. Have fun testing it out!')
                    ]
                )
            ]
        ),
        html.Div(
            className='row',
            children=[
                html.Div(
                    className='pretty_container ten columns offset-by-one columns',
                    children=[dcc.Input(
                        id="input_{}".format("text"), type="text",
                        placeholder="What is your Headline?",
                ),
                    ]
                ),
            ],
        ),
        html.Div(className='row',
                 children=[
                     html.Div(
                         className='pretty_container ten columns offset-by-one columns',
                         children=[
                             html.Div(id="out-all-types")
                         ]
                     )
                 ]),
        html.Div(className='row',
                 children=[
                     html.Img(
                         className='pretty_container ten columns offset-by-one columns',
                         src='https://img.nzz.ch/C=W4739,H2665.688,X0,Y247.1563/S=W1200M,H675M/O=75/C=AR1200x675/https://nzz-img.s3.amazonaws.com/2020/5/5/8c9dfe2a-1e93-4e08-9e46-f87b5677daa8.jpeg?wmark=nzz'
                     )
                 ]),

        #
    ]
)

@app.callback(
    Output("out-all-types", "children"),
    [Input("input_{}".format("text"), "value")],
)
def cb_render(input_string):
    url_trans = 'https://api.deepl.com/v2/translate'
    payload = {'auth_key': 'e90be2dd-92b3-920e-5d78-be332af77f0b',
               'text': input_string,
               'target_lang': 'EN'}
    r = requests.get(url_trans, params=payload)
    output_text = json.loads(r.text)['translations'][0]
    input_string = output_text['text']

    badofwords_validation = vectorizer.transform([str(input_string)])
    X_validation = badofwords_validation.toarray()
    predictions = model.predict(X_validation)

    if predictions[0] == 'b':
        predictions = 'Economics & Business'

    if predictions[0] == 't':
        predictions = 'Science & Technology'

    if predictions[0] == 'm':
        predictions = 'Health & Health Care'

    if predictions[0] == 'e':
        predictions = 'Entertainment'

    return predictions


if __name__ == "__main__":
    app.run_server(debug=False)
