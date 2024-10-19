import requests

from ..modules.c_stream.c_stream import gen_compatiblity_mode

class ChatAPI:
    def __init__(self, token, base, _legacy: bool = False):
        self.token = token
        self.base = base
        self.version = "2024-02-15-preview"

        self._legacy = _legacy

    def chat(
            self, 
            model: str,
            messages: list[dict[str, str]], 
            temperature: float = 0.7, 
            top_p: float = 1.0, 
            max_tokens: float = 150, 
            frequency_penalty: float = 1.0, 
            presence_penalty: float = 1.0, 
            stream: bool = False
        ):

        """
        This function sends a POST request to the API to generate a chat completion.

        :param model: The model name.
        :param messages: A list of messages.
        :param temperature: The temperature value.
        :param top_p: The top_p value.
        :param max_tokens: The max_tokens value.
        :param frequency_penalty: The frequency_penalty value.
        :param presence_penalty: The presence_penalty value.
        :param stream: The stream value.
        """
    

        self.headers = {
            "api-key": self.token,
            "Content-Type": "application/json"
        }

        self.endpoint = self.base + "/openai/deployments/" + model + "/chat/completions?api-version=" + self.version

        payload = {
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "stream": stream
        }

        try:
            response = requests.post(self.endpoint, headers=self.headers, json=payload, stream=stream)
            response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
                                                                  
            if stream:

                if not self._legacy:

                    for line in response.iter_lines():

                        if line:
        
                            yield line + b'\n\n'

                # Compatibility mode
                else:

                    for chunk in gen_compatiblity_mode(response, model=model):

                        yield chunk.decode()

            else:

                return response.content

        except requests.RequestException as e:
            if response.status_code == 401:               
                return {"error": "Invalid token."}
            
            elif response.status_code == 404:
                return {"error": f"Model {model} has not been deployed yet."}
            
            if response.status_code == 429:
                return {"error": "The current quota has been exceeded."}
            
            else:
                return {"error": f"An unexpected error occurred: {e}"}
