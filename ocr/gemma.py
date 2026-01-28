from google import genai
import time

class Gemma:
    def __init__(self, api_key: str, model='gemma-3-4b-it'):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.delay = 0
        self.last_response = None

    def generate_content(self, contents: list):
        """
          Generate content with given contents (image and prompt)

          Args:
            - contents: A list containing images and/or text prompts

          Returns:
            - Generated text response by Gemma
        """

        if contents is None or len(contents) == 0:
            raise ValueError("Contents must be provided.")

        self.delay -= time.time()

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents
        )

        self.delay += time.time()
        self.last_response = response

        return response.text
    
    def print_delay(self):
        """
          Prints the response time for the last generation request
        """
        print(f"Response Time: {self.delay: .3f} seconds")