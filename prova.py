#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flask import Flask, stream_with_context, request, Response, flash, stream_template
from time import sleep

app = Flask(__name__)


data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]


def generate():
    for item in data:
        yield str(item)
        sleep(1)


@app.route('/')
def stream_view():
    rows = generate()
    show_modal = modal()
    return Response(stream_template('template.html', rows=rows, show_modal=show_modal))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

