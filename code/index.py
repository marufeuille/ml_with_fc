# -*- coding: utf-8 -*-

import logging
import sklearn
import pickle
import json
import os
import urllib.request

def initializer(context):
    global lr
    url = os.environ["MODEL_URL"]
    response = urllib.request.urlopen(url)
    lr = pickle.loads(response.read())

def handler(environ, start_response):

    request_method = environ['REQUEST_METHOD']
    try:
        request_body_size = int(environ.get('CONTENT_LENGTH', 0))
        request_body = environ['wsgi.input'].read(request_body_size)
    except (ValueError):
        request_body_size = 0
    if request_method == 'POST':
        print(request_body.decode("utf-8"))
        js = json.loads(request_body.decode("utf-8"))
        X = [js["Time"]]
        for i in range(1, 29):
            X.append(js["V{}".format(i)])
        X.append(js["Amount"])
        preds = lr.predict_proba([X])
        response = str(preds[0,1]).encode("utf-8")
    else:
        response = b'Hello World!!'
    status = '200 OK'
    response_headers = [('Content-type', 'text/plain')]
    start_response(status, response_headers)
    return [response]

