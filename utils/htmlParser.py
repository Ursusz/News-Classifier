from bs4 import BeautifulSoup
import requests

def get_html(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        paras = soup.findAll('p')

        h1_element = soup.find('h1').get_text()

        paras_content = ''
        for para in paras:
            if para is not '':
                paras_content += para.get_text()

        if h1_element and paras:
            return h1_element, paras_content
        elif h1_element:
            return h1_element, ''
        elif paras_content:
            return '', paras_content
        else:
            return '', ''

    except requests.exceptions.RequestException as e:
        print(f"Error accessing URL {url}: {e}")
        return None, None
    except Exception as e:
        print(f"Error parsing URL {url}: {e}")
        return None, None