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
import newsapi
from newsapi import NewsApiClient

# read models
model = pickle.load(open("classifier.p", "rb"))
vectorizer = pickle.load(open("vectorizer.p", "rb"))


app = dash.Dash(__name__)
server = app.server
app.config.suppress_callback_exceptions = True

# pulling data from newsapi:
newsapi = NewsApiClient(api_key='4ffaa0cb22f44814800f9b47f3fc176e')  # client key should be secret

# pulling some news here
everything_german = newsapi.get_everything(q='corona', language='de')
df_in_german = pd.DataFrame(everything_german['articles'])

app.layout = html.Div(
    children=[
        html.H1(
            className='bg-grey',
            children=[
                'Welcome to ReadIT'
            ]
        ),
        html.Div(className='row',
                 children=[
                     html.Div(className='pretty_container four columns',
                              children=[
                                  html.Div(
                                      className='row',
                                      children=[html.Img(
                                          className='pretty_container twelve columns',
                                          src=df_in_german.urlToImage[0]
                                      )
                                      ]
                                  ),
                                  html.H5(
                                      className='row',
                                      children=[df_in_german.title[0]]
                                  ),
                                  html.H6(
                                      className='row',
                                      children=[df_in_german.description[0]]
                                  ),
                                  html.Div(className='bare_container four columns offset-by-eight columns',
                                           children=[
                                               html.A("ReadIT", href=df_in_german.url[0]),  # , target="_blank"
                                                   ]),
                              ]),
                     html.Div(className='pretty_container_two four columns',
                              children=[
                                  html.Div(
                                      className='row',
                                      children=[html.Img(
                                          className='pretty_container twelve columns',
                                          src=df_in_german.urlToImage[1]
                                      )
                                      ]
                                  ),
                                  html.H5(
                                      className='row',
                                      children=[df_in_german.title[1]]
                                  ),
                                  html.H6(
                                      className='row',
                                      children=[df_in_german.description[1]]
                                  ),
                                  html.Div(
                                      className='row',
                                      children=[
                                          html.Div(className='bare_container four columns offset-by-eight columns',
                                                   children=[
                                                       html.A("ReadIT", href=df_in_german.url[1]),  # , target="_blank"
                                                   ]),
                                      ]
                                  ),
                              ]),
                     html.Div(className='pretty_container four columns',
                              children=[
                                  html.Div(
                                      className='row',
                                      children=[html.Img(
                                          className='pretty_container twelve columns',
                                          src=df_in_german.urlToImage[2]
                                      )
                                      ]
                                  ),
                                  html.H5(
                                      className='row',
                                      children=[df_in_german.title[2]]
                                  ),
                                  html.H6(
                                      className='row',
                                      children=[df_in_german.description[2]]
                                  ),
                                  html.Div(
                                      className='row',
                                      children=[
                                          html.Div(className='bare_container four columns offset-by-eight columns',
                                                   children=[
                                                       html.A("ReadIT", href=df_in_german.url[2]),  # , target="_blank"
                                                   ]),
                                      ]
                                  ),
                              ])
                 ]),
        html.Div(className='row',
                 children=[
                     html.Div(className='pretty_container_two four columns',
                              children=[
                                  html.Div(
                                      className='row',
                                      children=[html.Img(
                                          className='pretty_container twelve columns',
                                          src=df_in_german.urlToImage[3]
                                      )
                                      ]
                                  ),
                                  html.H5(
                                      className='row',
                                      children=[df_in_german.title[3]]
                                  ),
                                  html.H6(
                                      className='row',
                                      children=[df_in_german.description[3]]
                                  ),
                                  html.Div(
                                      className='row',
                                      children=[
                                          html.Div(className='bare_container four columns offset-by-eight columns',
                                                   children=[
                                                       html.A("ReadIT", href=df_in_german.url[3]),  # , target="_blank"
                                                   ]),
                                      ]
                                  ),
                              ]),
                     html.Div(className='pretty_container four columns',
                              children=[
                                  html.Div(
                                      className='row',
                                      children=[html.Img(
                                          className='pretty_container twelve columns',
                                          src=df_in_german.urlToImage[4]
                                      )
                                      ]
                                  ),
                                  html.H5(
                                      className='row',
                                      children=[df_in_german.title[4]]
                                  ),
                                  html.H6(
                                      className='row',
                                      children=[df_in_german.description[4]]
                                  ),
                                  html.Div(
                                      className='row',
                                      children=[
                                          html.Div(className='bare_container four columns offset-by-eight columns',
                                                   children=[
                                                       html.A("ReadIT", href=df_in_german.url[4]),  # , target="_blank"
                                                   ]),
                                      ]
                                  ),
                              ]),
                     html.Div(className='pretty_container_two four columns',
                              children=[
                                  html.Div(
                                      className='row',
                                      children=[html.Img(
                                          className='pretty_container twelve columns',
                                          src=df_in_german.urlToImage[5]
                                      )
                                      ]
                                  ),
                                  html.H5(
                                      className='row',
                                      children=[df_in_german.title[5]]
                                  ),
                                  html.H6(
                                      className='row',
                                      children=[df_in_german.description[5]]
                                  ),
                                  html.Div(
                                      className='row',
                                      children=[
                                          html.Div(className='bare_container four columns offset-by-eight columns',
                                                   children=[
                                                       html.A("ReadIT", href=df_in_german.url[5]),  # , target="_blank"
                                                   ]),
                                      ]
                                  ),
                              ]),
                         ]),
        html.Div(className='row',
                         children=[
                             html.Div(className='pretty_container four columns',
                                      children=[
                                          html.Div(
                                              className='row',
                                              children=[html.Img(
                                                  className='pretty_container twelve columns',
                                                  src=df_in_german.urlToImage[6]
                                              )
                                              ]
                                          ),
                                          html.H5(
                                              className='row',
                                              children=[df_in_german.title[6]]
                                          ),
                                          html.H6(
                                              className='row',
                                              children=[df_in_german.description[6]]
                                          ),
                                          html.Div(
                                      className='row',
                                      children=[
                                          html.Div(className='bare_container four columns offset-by-eight columns',
                                                   children=[
                                                       html.A("ReadIT", href=df_in_german.url[6]),  # , target="_blank"
                                                   ]),
                                      ]
                                  ),
                                      ]),
                             html.Div(className='pretty_container_two four columns',
                                      children=[
                                          html.Div(
                                              className='row',
                                              children=[html.Img(
                                                  className='pretty_container twelve columns',
                                                  src=df_in_german.urlToImage[7]
                                              )
                                              ]
                                          ),
                                          html.H5(
                                              className='row',
                                              children=[df_in_german.title[7]]
                                          ),
                                          html.H6(
                                              className='row',
                                              children=[df_in_german.description[7]]
                                          ),
                                          html.Div(
                                      className='row',
                                      children=[
                                          html.Div(className='bare_container four columns offset-by-eight columns',
                                                   children=[
                                                       html.A("ReadIT", href=df_in_german.url[7]),  # , target="_blank"
                                                   ]),
                                      ]
                                  ),
                                      ]),
                             html.Div(className='pretty_container four columns',
                                      children=[
                                          html.Div(
                                              className='row',
                                              children=[html.Img(
                                                  className='pretty_container twelve columns',
                                                  src=df_in_german.urlToImage[8]
                                              )
                                              ]
                                          ),
                                          html.H5(
                                              className='row',
                                              children=[df_in_german.title[8]]
                                          ),
                                          html.H6(
                                              className='row',
                                              children=[df_in_german.description[8]]
                                          ),
                                          html.Div(
                                      className='row',
                                      children=[
                                          html.Div(className='bare_container four columns offset-by-eight columns',
                                                   children=[
                                                       html.A("ReadIT", href=df_in_german.url[8]),  # , target="_blank"
                                                   ]),
                                      ]
                                  ),
                                      ])
                         ]),
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
