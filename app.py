import logging
import os
import sys

# no pycache
sys.dont_write_bytecode = True

import json

from flask import       Flask, request, jsonify, Response, stream_with_context
from flask_cors import  CORS

from assets.API.Inference import ChatAPI

# make logger
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# create a Flask app
app = Flask(__name__)

# read from the config file
try:
    with open("config.json", "r") as cfile:
        config = json.load(cfile)
except FileNotFoundError:
    config = {}

@app.route("/v1/models", methods=["GET"])
def models():
    return jsonify({
        'data': [
            {"id": "gpt-4", "name": "gpt-4-turbo-2024-0409", "context": 128000},
            {"id": "gpt-4o", "name": "gpt-4o", "context": 128000},
        ]
    })

@app.route("/v1/chat/completions", methods=["POST"])
def chat():
    # Get 'Authorization' or 'api-key' from headers
    token = request.headers.get("Authorization") or request.headers.get("api-key")
    if not token:
        return jsonify({"error": "You are not authorized to visit this webpage."}), 401

    if "Bearer " in token:
        token = token.split(" ")[1]

    # Extract request data
    data = request.json
    model = data["model"]
    messages = data["messages"]
    temperature = data.get("temperature", 0.7)
    top_p = data.get("top_p", 1.0)
    max_tokens = data.get("max_completion_tokens", data.get("max_tokens", 150))
    frequency_penalty = data.get("frequency_penalty", 1.0)
    presence_penalty = data.get("presence_penalty", 1.0)
    stream = data.get("stream", False)

    # Initialize API
    api = ChatAPI(token, os.environ['BASE_URL'], _legacy=config.get("COMPAT_MODE", False))

    @stream_with_context
    def return_stream():

        for line in api.chat(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stream=stream,
        ):
            if line:

                if "error" in line:
                    return jsonify({"error": "You have probably hit a filter"}), 400

                yield line

    if stream:

        return Response(
            return_stream(),
            status=200,
            content_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Transfer-Encoding": "chunked"
            },
        )
    
    chatCompletion_nos = api.chat(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stream=stream,
    )

    # If not streaming, return as a JSON response
    return jsonify(chatCompletion_nos)
    

@app.errorhandler(404)
def page_not_found(e):
    logging.error("404 error")
    return jsonify({"error": "The requested URL was not found on the server."}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    logging.error("405 error")
    return jsonify({"error": "The method is not allowed for the requested URL."}), 405

@app.errorhandler(500)
def internal_server_error(e):
    logging.error("500 error")
    return jsonify({"error": "The server encountered an internal error."}), 500

if __name__ == "__main__":

    CORS(app)

    # check if we need to create a global cloudflare tunnel
    if config.get("GLOBALIZE"):

        # import expose model
        try:
            from assets.cloudflare.Expose import create_cloudflare_tunnel
        except ImportError: 
            print("Cloudflare tunnel module not found")
            exit(1)

        # create a cloudflare tunnel
        #create_cloudflare_tunnel(config.get("PORT", 5000))

    app.run(
        host=config.get("HOST", "0.0.0.0"),
        port=config.get("PORT", 5000),
        debug=config.get("DEBUG", False),
        threaded=config.get("THREADED", True)
    )
