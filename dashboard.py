import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import joblib
import os
import pickle
import requests
import json
import pandas as pd
import numpy as np
import newsapi
from newsapi import NewsApiClient

# read models
model = pickle.load(open("classifier.p", "rb"))
vectorizer = pickle.load(open("vectorizer.p", "rb"))


app = dash.Dash(__name__)
server = app.server
app.config.suppress_callback_exceptions = True

# sui: 9e986f7fac504c11869e5068228f4ae7
# uni: 7c0d305c05bd4c8fbd1c978c34e2613e
# hotmail: 66f91311df8a482bb4ac74a7c96e373c
# gmail: 4ffaa0cb22f44814800f9b47f3fc176e

# pulling data from newsapi:
newsapi = NewsApiClient(
    api_key="9e986f7fac504c11869e5068228f4ae7"
)  # client key should be secret

# pulling some news here
everything_german = newsapi.get_everything(
    sources="handelsblatt,wirtschafts-woche,the-washington-post,the-wall-street-journal"
    # q='Flüchtling',
    # language='de',
)
df_in_german = pd.DataFrame(everything_german["articles"])

sources = newsapi.get_sources()
df_sources = pd.DataFrame(sources["sources"])


def get_options(list_stocks):
    dict_list = []
    for i in list_stocks:
        dict_list.append({"label": i, "value": i})

    return dict_list


app.layout = html.Div(
    children=[
        html.Div(
            className="bare_container ten columns offset-by-one column",
            children=[
                dcc.Tabs(
                    className="bare_container eight columns offset-by-four column",
                    children=[
                        dcc.Tab(
                            className="custom-tab",
                            selected_className="custom-tab--selected",
                            label="NEWS ROOM",
                            children=[
                                html.Div(
                                    className="one-third column",
                                    children=[
                                        html.Img(
                                            className="border_container twelve columns",
                                            src="https://raw.githubusercontent.com/nickzumbuehl/read_IT/master/logo.png",
                                            style={'height':'50%', 'width':'50%'}
                                        ),
                                        html.Div(
                                            className="pretty_container twelve columns",
                                            children=[
                                                html.Div(
                                                    className="row",
                                                    children=[
                                                        html.Img(
                                                            className="pretty_container twelve columns",
                                                            src=df_in_german.urlToImage[
                                                                0
                                                            ],
                                                        )
                                                    ],
                                                ),
                                                html.H5(
                                                    className="row",
                                                    children=[df_in_german.title[0]],
                                                ),
                                                html.H6(
                                                    className="row",
                                                    children=[
                                                        df_in_german.description[0]
                                                        + " (Source: {})".format(
                                                            df_in_german.source[0][
                                                                "name"
                                                            ]
                                                        )
                                                    ],
                                                ),
                                                html.Div(
                                                    className="row",
                                                    children=[
                                                        html.Div(
                                                            className="bare_container four columns offset-by-eight columns",
                                                            children=[
                                                                html.A(
                                                                    "ReadIT",
                                                                    href=df_in_german.url[
                                                                        0
                                                                    ],
                                                                    target="_blank",
                                                                )
                                                            ],
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="pretty_container_two twelve columns",
                                            children=[
                                                html.Div(
                                                    className="row",
                                                    children=[
                                                        html.Img(
                                                            className="pretty_container_two twelve columns",
                                                            src=df_in_german.urlToImage[
                                                                1
                                                            ],
                                                        )
                                                    ],
                                                ),
                                                html.H5(
                                                    className="row",
                                                    children=[df_in_german.title[1]],
                                                ),
                                                html.H6(
                                                    className="row",
                                                    children=[
                                                        df_in_german.description[1]
                                                        + " (Source: {})".format(
                                                            df_in_german.source[1][
                                                                "name"
                                                            ]
                                                        )
                                                    ],
                                                ),
                                                html.Div(
                                                    className="row",
                                                    children=[
                                                        html.Div(
                                                            className="bare_container four columns offset-by-eight columns",
                                                            children=[
                                                                html.A(
                                                                    "ReadIT",
                                                                    href=df_in_german.url[
                                                                        1
                                                                    ],
                                                                    target="_blank",
                                                                )
                                                            ],
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="pretty_container twelve columns",
                                            children=[
                                                html.Div(
                                                    className="row",
                                                    children=[
                                                        html.Img(
                                                            className="pretty_container twelve columns",
                                                            src=df_in_german.urlToImage[
                                                                2
                                                            ],
                                                        )
                                                    ],
                                                ),
                                                html.H5(
                                                    className="row",
                                                    children=[df_in_german.title[2]],
                                                ),
                                                html.H6(
                                                    className="row",
                                                    children=[
                                                        df_in_german.description[2]
                                                        + " (Source: {})".format(
                                                            df_in_german.source[2][
                                                                "name"
                                                            ]
                                                        )
                                                    ],
                                                ),
                                                html.Div(
                                                    className="row",
                                                    children=[
                                                        html.Div(
                                                            className="bare_container four columns offset-by-eight columns",
                                                            children=[
                                                                html.A(
                                                                    "ReadIT",
                                                                    href=df_in_german.url[
                                                                        2
                                                                    ],
                                                                    target="_blank",
                                                                )
                                                            ],
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="one-third column",
                                    children=[
                                        html.Div(
                                            className="pretty_container_two twelve columns",
                                            children=[
                                                html.Div(
                                                    className="row",
                                                    children=[
                                                        html.Img(
                                                            className="pretty_container_two twelve columns",
                                                            src=df_in_german.urlToImage[
                                                                3
                                                            ],
                                                        )
                                                    ],
                                                ),
                                                html.H5(
                                                    className="row",
                                                    children=[df_in_german.title[3]],
                                                ),
                                                html.H6(
                                                    className="row",
                                                    children=[
                                                        df_in_german.description[3]
                                                        + " (Source: {})".format(
                                                            df_in_german.source[3][
                                                                "name"
                                                            ]
                                                        )
                                                    ],
                                                ),
                                                html.Div(
                                                    className="row",
                                                    children=[
                                                        html.Div(
                                                            className="bare_container four columns offset-by-eight columns",
                                                            children=[
                                                                html.A(
                                                                    "ReadIT",
                                                                    href=df_in_german.url[
                                                                        3
                                                                    ],
                                                                    target="_blank",
                                                                )
                                                            ],
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="pretty_container twelve columns",
                                            children=[
                                                html.Div(
                                                    className="row",
                                                    children=[
                                                        html.Img(
                                                            className="pretty_container twelve columns",
                                                            src=df_in_german.urlToImage[
                                                                4
                                                            ],
                                                        )
                                                    ],
                                                ),
                                                html.H5(
                                                    className="row",
                                                    children=[df_in_german.title[4]],
                                                ),
                                                html.H6(
                                                    className="row",
                                                    children=[
                                                        df_in_german.description[4]
                                                        + " (Source: {})".format(
                                                            df_in_german.source[4][
                                                                "name"
                                                            ]
                                                        )
                                                    ],
                                                ),
                                                html.Div(
                                                    className="row",
                                                    children=[
                                                        html.Div(
                                                            className="bare_container four columns offset-by-eight columns",
                                                            children=[
                                                                html.A(
                                                                    "ReadIT",
                                                                    href=df_in_german.url[
                                                                        4
                                                                    ],
                                                                    target="_blank",
                                                                )
                                                            ],
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="pretty_container_two twelve columns",
                                            children=[
                                                html.Div(
                                                    className="row",
                                                    children=[
                                                        html.Img(
                                                            className="pretty_container_two twelve columns",
                                                            src=df_in_german.urlToImage[
                                                                5
                                                            ],
                                                        )
                                                    ],
                                                ),
                                                html.H5(
                                                    className="row",
                                                    children=[df_in_german.title[5]],
                                                ),
                                                html.H6(
                                                    className="row",
                                                    children=[
                                                        df_in_german.description[5]
                                                        + " (Source: {})".format(
                                                            df_in_german.source[5][
                                                                "name"
                                                            ]
                                                        )
                                                    ],
                                                ),
                                                html.Div(
                                                    className="row",
                                                    children=[
                                                        html.Div(
                                                            className="bare_container four columns offset-by-eight columns",
                                                            children=[
                                                                html.A(
                                                                    "ReadIT",
                                                                    href=df_in_german.url[
                                                                        5
                                                                    ],
                                                                    target="_blank",
                                                                )
                                                            ],
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="one-third column",
                                    children=[
                                        html.Div(
                                            className="pretty_container twelve columns",
                                            children=[
                                                html.Div(
                                                    className="row",
                                                    children=[
                                                        html.Img(
                                                            className="pretty_container twelve columns",
                                                            src=df_in_german.urlToImage[
                                                                6
                                                            ],
                                                        )
                                                    ],
                                                ),
                                                html.H5(
                                                    className="row",
                                                    children=[df_in_german.title[6]],
                                                ),
                                                html.H6(
                                                    className="row",
                                                    children=[
                                                        df_in_german.description[6]
                                                        + " (Source: {})".format(
                                                            df_in_german.source[6][
                                                                "name"
                                                            ]
                                                        )
                                                    ],
                                                ),
                                                html.Div(
                                                    className="row",
                                                    children=[
                                                        html.Div(
                                                            className="bare_container four columns offset-by-eight columns",
                                                            children=[
                                                                html.A(
                                                                    "ReadIT",
                                                                    href=df_in_german.url[
                                                                        6
                                                                    ],
                                                                    target="_blank",
                                                                )
                                                            ],
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="pretty_container_two twelve columns",
                                            children=[
                                                html.Div(
                                                    className="row",
                                                    children=[
                                                        html.Img(
                                                            className="pretty_container_two twelve columns",
                                                            src=df_in_german.urlToImage[
                                                                7
                                                            ],
                                                        )
                                                    ],
                                                ),
                                                html.H5(
                                                    className="row",
                                                    children=[df_in_german.title[7]],
                                                ),
                                                html.H6(
                                                    className="row",
                                                    children=[
                                                        df_in_german.description[7]
                                                        + " (Source: {})".format(
                                                            df_in_german.source[7][
                                                                "name"
                                                            ]
                                                        )
                                                    ],
                                                ),
                                                html.Div(
                                                    className="row",
                                                    children=[
                                                        html.Div(
                                                            className="bare_container four columns offset-by-eight columns",
                                                            children=[
                                                                html.A(
                                                                    "ReadIT",
                                                                    href=df_in_german.url[
                                                                        7
                                                                    ],
                                                                    target="_blank",
                                                                )
                                                            ],
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="pretty_container twelve columns",
                                            children=[
                                                html.Div(
                                                    className="row",
                                                    children=[
                                                        html.Img(
                                                            className="pretty_container twelve columns",
                                                            src=df_in_german.urlToImage[
                                                                8
                                                            ],
                                                        )
                                                    ],
                                                ),
                                                html.H5(
                                                    className="row",
                                                    children=[df_in_german.title[8]],
                                                ),
                                                html.H6(
                                                    className="row",
                                                    children=[
                                                        df_in_german.description[8]
                                                        + " (Source: {})".format(
                                                            df_in_german.source[8][
                                                                "name"
                                                            ]
                                                        )
                                                    ],
                                                ),
                                                html.Div(
                                                    className="row",
                                                    children=[
                                                        html.Div(
                                                            className="bare_container four columns offset-by-eight columns",
                                                            children=[
                                                                html.A(
                                                                    "ReadIT",
                                                                    href=df_in_german.url[
                                                                        8
                                                                    ],
                                                                    target="_blank",
                                                                )
                                                            ],
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
                                        html.Img(
                                            className="border_container twelve columns",
                                            src="https://raw.githubusercontent.com/nickzumbuehl/read_IT/master/logo.png",
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        dcc.Tab(
                            className="custom-tab",
                            selected_className="custom-tab--selected",
                            label="FILTER ENGINE",
                            children=[
                                html.Div(
                                    className="bare_container twelve columns",
                                    children=[
                                        html.Div(
                                            className="border_container one-third column",  # row
                                            children=[
                                                html.Img(
                                                    className="bare_container twelve columns",
                                                    src="https://raw.githubusercontent.com/nickzumbuehl/read_IT/master/logo.png",
                                                    style={'height':'50%', 'width':'50%'}
                                                ),
                                                html.Div(
                                                    className="twelve columns",
                                                    children=[
                                                        html.Div(
                                                            className="pretty_container twelve columns",
                                                            children=[
                                                                dcc.Dropdown(
                                                                    id="sources",
                                                                    options=get_options(
                                                                        df_sources.id
                                                                    ),
                                                                    multi=False,
                                                                )
                                                            ],
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    className="twelve columns",
                                                    children=[
                                                        html.Div(
                                                            className="pretty_container_two twelve columns",
                                                            children=[
                                                                dcc.Dropdown(
                                                                    id="language",
                                                                    options=get_options(
                                                                        df_sources.language.unique()
                                                                    ),
                                                                    multi=False,
                                                                    value="en",
                                                                )
                                                            ],
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    className="twelve columns",
                                                    children=[
                                                        html.Div(
                                                            className="pretty_container twelve columns",
                                                            children=[
                                                                dcc.Input(
                                                                    id="word_search",
                                                                    placeholder="Key word?",
                                                                    value="oil AND euro",
                                                                )
                                                            ],
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    className="twelve columns",
                                                    children=[
                                                        html.Div(
                                                            className="pretty_container_two twelve columns",
                                                            children=[
                                                                html.Button(
                                                                    'Filter IT',
                                                                    id="filter_button",
                                                                    type='submit',
                                                                )
                                                            ],
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="row",
                                            children=[html.Div(id="output")],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        dcc.Tab(
                            className="custom-tab",
                            selected_className="custom-tab--selected",
                            label="VISIT THE SOURCE",
                        ),
                    ],
                ),
                html.H6(),
            ],
        ),
        # html.H1(
        #     className='row bg-grey',
        #     children=[
        #         'AI Classification of Unseen News Headlines'
        #     ]
        # ),
        # html.Div(
        #     className='row',
        #     children=[
        #         html.Div(
        #             className='pretty_container ten columns offset-by-one columns',
        #             children=[
        #                 html.H2('The algorithm was trained on a data set consisting of Economics & Business,'
        #                         ' Science & Technology, Health & Health Care and Entertainment news articles from the'
        #                         ' United States. The AI aims to classify unseen news headlines according to these'
        #                         ' four clusters. Have fun testing it out!')
        #             ]
        #         )
        #     ]
        # ),
        # html.Div(
        #     className='row',
        #     children=[
        #         html.Div(
        #             className='pretty_container ten columns offset-by-one columns',
        #             children=[dcc.Input(
        #                 id="input_{}".format("text"), type="text",
        #                 placeholder="What is your Headline?",
        #             ),
        #             ]
        #         ),
        #     ],
        # ),
        # html.Div(className='row',
        #          children=[
        #              html.Div(
        #                  className='pretty_container ten columns offset-by-one columns',
        #                  children=[
        #                      html.Div(id="out-all-types")
        #                  ]
        #              )
        #          ]),
    ]
)


@app.callback(
    Output("output", "children"),
    [
        # Input("sources", "value"),
        # Input("language", "value"),
        # Input("word_search", "value"),
        Input('filter_button', "n_clicks"),
    ],
    [
        State('sources', 'value'),
        State('language', 'value'),
        State('word_search', 'value')
    ]
)
def generate_output(n_clicks_button, input_sources, input_languages, input_word_search):
    everything_selected = newsapi.get_everything(
        sources=input_sources, q=input_word_search, language=input_languages,
    )

    df_ = pd.DataFrame(everything_selected["articles"])

    out = html.Div(
        children=[
                html.Div(
                    className="one-third column",
                    children=[
                        html.Div(
                            className="pretty_container twelve columns",
                            children=[
                                html.Div(
                                    className="row",
                                    children=[
                                        html.Img(
                                            className="pretty_container twelve columns",
                                            src=df_.urlToImage[0],
                                        )
                                    ],
                                ),
                                html.H5(className="row", children=[df_.title[0]],),
                                html.H6(
                                    className="row",
                                    children=[
                                        df_.description[0]
                                        + " (Source: {})".format(df_.source[0]["name"])
                                    ],
                                ),
                                html.Div(
                                    className="row",
                                    children=[
                                        html.Div(
                                            className="bare_container four columns offset-by-eight columns",
                                            children=[
                                                html.A(
                                                    "ReadIT",
                                                    href=df_.url[0],
                                                    target="_blank",
                                                )
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        html.Div(
                            className="pretty_container_two twelve columns",
                            children=[
                                html.Div(
                                    className="row",
                                    children=[
                                        html.Img(
                                            className="pretty_container_two twelve columns",
                                            src=df_.urlToImage[1],
                                        )
                                    ],
                                ),
                                html.H5(className="row", children=[df_.title[1]],),
                                html.H6(
                                    className="row",
                                    children=[
                                        df_.description[1]
                                        + " (Source: {})".format(df_.source[1]["name"])
                                    ],
                                ),
                                html.Div(
                                    className="row",
                                    children=[
                                        html.Div(
                                            className="bare_container four columns offset-by-eight columns",
                                            children=[
                                                html.A(
                                                    "ReadIT",
                                                    href=df_.url[1],
                                                    target="_blank",
                                                )
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        html.Div(
                            className="pretty_container twelve columns",
                            children=[
                                html.Div(
                                    className="row",
                                    children=[
                                        html.Img(
                                            className="pretty_container twelve columns",
                                            src=df_.urlToImage[2],
                                        )
                                    ],
                                ),
                                html.H5(className="row", children=[df_.title[2]],),
                                html.H6(
                                    className="row",
                                    children=[
                                        df_.description[2]
                                        + " (Source: {})".format(df_.source[2]["name"])
                                    ],
                                ),
                                html.Div(
                                    className="row",
                                    children=[
                                        html.Div(
                                            className="bare_container four columns offset-by-eight columns",
                                            children=[
                                                html.A(
                                                    "ReadIT",
                                                    href=df_.url[2],
                                                    target="_blank",
                                                )
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
                html.Div(
                    className="one-third column",
                    children=[
                        html.Div(
                            className="pretty_container_two twelve columns",
                            children=[
                                html.Div(
                                    className="row",
                                    children=[
                                        html.Img(
                                            className="pretty_container_two twelve columns",
                                            src=df_.urlToImage[3],
                                        )
                                    ],
                                ),
                                html.H5(className="row", children=[df_.title[3]],),
                                html.H6(
                                    className="row",
                                    children=[
                                        df_.description[3]
                                        + " (Source: {})".format(df_.source[3]["name"])
                                    ],
                                ),
                                html.Div(
                                    className="row",
                                    children=[
                                        html.Div(
                                            className="bare_container four columns offset-by-eight columns",
                                            children=[
                                                html.A(
                                                    "ReadIT",
                                                    href=df_.url[3],
                                                    target="_blank",
                                                )
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        html.Div(
                            className="pretty_container twelve columns",
                            children=[
                                html.Div(
                                    className="row",
                                    children=[
                                        html.Img(
                                            className="pretty_container twelve columns",
                                            src=df_.urlToImage[4],
                                        )
                                    ],
                                ),
                                html.H5(className="row", children=[df_.title[4]],),
                                html.H6(
                                    className="row",
                                    children=[
                                        df_.description[4]
                                        + " (Source: {})".format(df_.source[4]["name"])
                                    ],
                                ),
                                html.Div(
                                    className="row",
                                    children=[
                                        html.Div(
                                            className="bare_container four columns offset-by-eight columns",
                                            children=[
                                                html.A(
                                                    "ReadIT",
                                                    href=df_.url[4],
                                                    target="_blank",
                                                )
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        html.Div(
                            className="pretty_container_two twelve columns",
                            children=[
                                html.Div(
                                    className="row",
                                    children=[
                                        html.Img(
                                            className="pretty_container_two twelve columns",
                                            src=df_.urlToImage[5],
                                        )
                                    ],
                                ),
                                html.H5(className="row", children=[df_.title[5]],),
                                html.H6(
                                    className="row",
                                    children=[
                                        df_.description[5]
                                        + " (Source: {})".format(df_.source[5]["name"])
                                    ],
                                ),
                                html.Div(
                                    className="row",
                                    children=[
                                        html.Div(
                                            className="bare_container four columns offset-by-eight columns",
                                            children=[
                                                html.A(
                                                    "ReadIT",
                                                    href=df_.url[5],
                                                    target="_blank",
                                                )
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ]
    )

    return out



@app.callback(
    Output("out-all-types", "children"), [Input("input_{}".format("text"), "value")],
)
def cb_render(input_string):
    url_trans = "https://api.deepl.com/v2/translate"
    payload = {
        "auth_key": "e90be2dd-92b3-920e-5d78-be332af77f0b",
        "text": input_string,
        "target_lang": "EN",
    }
    r = requests.get(url_trans, params=payload)
    output_text = json.loads(r.text)["translations"][0]
    input_string = output_text["text"]

    badofwords_validation = vectorizer.transform([str(input_string)])
    X_validation = badofwords_validation.toarray()
    predictions = model.predict(X_validation)

    if predictions[0] == "b":
        predictions = "Economics & Business"

    if predictions[0] == "t":
        predictions = "Science & Technology"

    if predictions[0] == "m":
        predictions = "Health & Health Care"

    if predictions[0] == "e":
        predictions = "Entertainment"

    return predictions


if __name__ == "__main__":
    app.run_server(debug=False)
