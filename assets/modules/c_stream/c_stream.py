import json
from secrets import     token_hex
from time import        time
from typing import      Generator, Iterable

def gen_compatiblity_mode(response: Iterable[bytes], model: str) -> Generator[bytes, None, None]:

    """
    This function generates a new streaming dict that is compatible with the legacy front-ends.
    
    :param response: The response object from the API.
    :param model: The model name.
    
    :return: A generator object that yields the new streaming dict.
    """

    # JSON correctness mode

    __id: str = "chatcmpl-" + token_hex(14)
    __time: str = str(int(time()))

    for line in response.iter_lines():

        if line:

            # try to get raw content of the line
            try:

                __content = json.loads(str(line.decode().removeprefix("data: ")))["choices"][0]["delta"]["content"]

                # put it in a more minimalistic type of dict
                yield b'data: ' + json.dumps(
                    {
                        "id": __id,
                        "created": __time,
                        "model": model,
                        "object": "chat.completion.chunk",
                        "choices": [
                            {
                                "delta": {
                                    "content": __content
                                },
                                "index": 0
                            }
                        ],
                        "finish_reason": None
                    }
                ).encode() + b'\n\n'

            except (json.JSONDecodeError, KeyError, IndexError):

                continue

    # for the last two chunks, we need to yield a finish reason stop
    yield b'data: ' + json.dumps({
                "id": __id,
                "created": __time,
                "model": model,
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "delta": {
                            "content": ""
                        },
                        "index": 0
                    }
                ],
                "finish_reason": "stop"
            }
        ).encode() + b'\n\n'
            
    yield b'data: [DONE]'
