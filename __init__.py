import requests
import json
import re
import urllib.parse
from typing import Optional, Dict, Any, Union

class CivitaiAPI:
    BASE_URL = "https://civitai.com/api/v1"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_headers = {'Content-Type': 'application/json'}
        if api_key:
            self.base_headers["Authorization"] = f"Bearer {api_key}"

    def _request(self, method: str, endpoint: str) -> Union[Dict[str, Any], None]:
        url = f"{self.BASE_URL}/{endpoint.lstrip('/')}"
        try:
            response = requests.request(method, url, headers=self.base_headers, timeout=30)
            response.raise_for_status()
            if response.status_code == 204 or not response.content:
                return None
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            print(f"--> Erro de API do Civitai: {http_err} (URL: {url})")
            return {"error": str(http_err), "status_code": http_err.response.status_code}
        except requests.exceptions.RequestException as req_err:
            print(f"--> Erro de Requisição da API do Civitai: {req_err}")
            return {"error": str(req_err)}
        return None

    def get_model_info(self, model_id: int) -> Optional[Dict[str, Any]]:
        return self._request("GET", f"/models/{model_id}")

    def get_model_version_info(self, version_id: int) -> Optional[Dict[str, Any]]:
        return self._request("GET", f"/model-versions/{version_id}")

def parse_civitai_input(url_or_id: str) -> tuple[int | None, int | None]:
    url_or_id = str(url_or_id).strip()
    if not url_or_id:
        return None, None
        
    # Procura pelo padrão 'civitai:ID_MODELO@ID_VERSAO' ou 'urn:air:sdxl:lora:civitai:ID_MODELO@ID_VERSAO'
    air_match = re.search(r'(?:urn:air:sdxl:lora:)?(?:civitai:)?(\d+)@(\d+)', url_or_id)
    if air_match:
        model_id_str, version_id_str = air_match.groups()
        return int(model_id_str), int(version_id_str)

    if url_or_id.isdigit():
        return int(url_or_id), None

    try:
        parsed_url = urllib.parse.urlparse(url_or_id)
        path_parts = [p for p in parsed_url.path.split('/') if p]
        query_params = urllib.parse.parse_qs(parsed_url.query)

        model_id: int | None = None
        version_id: int | None = None

        if 'modelVersionId' in query_params:
            version_id = int(query_params['modelVersionId'][0])

        if "models" in path_parts:
            model_index = path_parts.index("models")
            if model_index + 1 < len(path_parts) and path_parts[model_index + 1].isdigit():
                model_id = int(path_parts[model_index + 1])

        if not version_id and "model-versions" in path_parts:
            version_index = path_parts.index("model-versions")
            if version_index + 1 < len(path_parts) and path_parts[version_index + 1].isdigit():
                version_id = int(path_parts[version_index + 1])
        
        return model_id, version_id
    except Exception:
        return None, None

def get_civitai_details(link_ou_id: str) -> Dict[str, Any]:
    """Extrai detalhes do CivitAI e retorna informações estruturadas"""
    model_id, version_id = parse_civitai_input(link_ou_id)

    if not model_id and not version_id:
        return {"error": "Não foi possível extrair um ID de modelo ou versão válido da entrada."}

    api = CivitaiAPI()
    
    try:
        if not model_id and version_id:
            temp_version_info = api.get_model_version_info(version_id)
            if temp_version_info and 'modelId' in temp_version_info:
                model_id = temp_version_info['modelId']
            else:
                return {"error": f"Não foi possível encontrar o modelo para a versão ID {version_id}."}

        model_info = api.get_model_info(model_id)
        if not model_info or "error" in model_info:
            return {"error": f"Falha ao buscar informações do modelo ID {model_id}."}

        if not version_id:
            if model_info.get("modelVersions") and len(model_info["modelVersions"]) > 0:
                version_id = model_info["modelVersions"][0].get('id')
                if not version_id:
                    return {"error": "Não foi possível encontrar o ID na versão mais recente."}
            else:
                return {"error": "O modelo não tem nenhuma versão listada."}

        version_info = api.get_model_version_info(version_id)
        if not version_info or "error" in version_info:
            return {"error": f"Falha ao buscar informações da versão ID {version_id}."}

    except Exception as e:
        return {"error": f"Erro ao processar: {str(e)}"}

    model_name = model_info.get('name', 'N/A')
    version_name = version_info.get('name', 'N/A')
    
    description_html = model_info.get('description')
    description_text = re.sub('<[^<]+?>', '', description_html).strip() if description_html else "Nenhuma descrição fornecida."
    trained_words = version_info.get('trainedWords', [])

    return {
        "model_name": model_name,
        "version_name": version_name,
        "model_id": model_id,
        "version_id": version_id,
        "air_id": f"{model_id}@{version_id}",
        "description": description_text,
        "trained_words": trained_words
    }

def call_openrouter_api(system_prompt: str, user_content: str, api_key: str) -> Dict[str, Any]:
    """Chama a API do OpenRouter com o modelo Qwen"""
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "ComfyUI-CivitAI-Processor",
                "X-Title": "ComfyUI CivitAI Processor",
            },
            data=json.dumps({
                "model": "qwen/qwen-2.5-72b-instruct",
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_content
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 2048
            }),
            timeout=60
        )
        
        response.raise_for_status()
        result = response.json()
        
        if 'choices' in result and len(result['choices']) > 0:
            content = result['choices'][0]['message']['content']
            try:
                # Tenta extrair JSON do conteúdo
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                else:
                    return {"error": "Resposta da LLM não contém JSON válido", "raw_response": content}
            except json.JSONDecodeError:
                return {"error": "Erro ao decodificar JSON da resposta", "raw_response": content}
        else:
            return {"error": "Resposta inesperada da API", "raw_response": result}
            
    except requests.exceptions.RequestException as e:
        return {"error": f"Erro na requisição: {str(e)}"}
    except Exception as e:
        return {"error": f"Erro inesperado: {str(e)}"}

class CivitAIInfoProcessor:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "link_or_id": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Cole aqui o link do CivitAI, ID do modelo ou AIR ID..."
                }),
                "openrouter_token": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "Seu token da OpenRouter API..."
                }),
            },
            "optional": {
                "additional_info": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Informações adicionais sobre o LORA ou execução (opcional)..."
                }),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Seu system prompt personalizado..."
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("lora_air_id", "character_name", "character_description", "s1_outfits", "s2_outfit", "s3_outfit")
    
    FUNCTION = "process_civitai_info"
    CATEGORY = "CivitAI"

    def process_civitai_info(self, link_or_id, openrouter_token, additional_info="", system_prompt=""):
        # Extrai informações do CivitAI
        civitai_data = get_civitai_details(link_or_id)
        
        if "error" in civitai_data:
            error_msg = civitai_data["error"]
            return (error_msg, error_msg, error_msg, error_msg, error_msg, error_msg)
        
        # Prepara o conteúdo para a LLM
        user_content = f"""USER INFO: {additional_info}
Modelo: {civitai_data['model_name']}
Versão: {civitai_data['version_name']}
ID AIR: civitai:{civitai_data['air_id']}
----------------------------------------
Descrição:
{civitai_data['description']}
----------------------------------------
Palavras de Gatilho:
{chr(10).join('- ' + word for word in civitai_data['trained_words'])}"""

        # Usa system prompt padrão se não fornecido
        if not system_prompt.strip():
            system_prompt = """Você é um assistente especializado em processar informações de modelos LORA do CivitAI. 
            Analise as informações fornecidas e extraia dados estruturados sobre personagens e suas roupas.
            
            Retorne APENAS um JSON válido com as seguintes chaves:
            - character_name: nome simples do personagem em lowercase com underscores
            - character_description: características físicas básicas (sem "solo", sem roupas)
            - s1: todas as roupas separadas por /cut, com versão sem sapatos quando aplicável
            - s2: roupa principal /cut _tokenClothing /cut _tokenClothing2
            - s3: roupa principal /cut _tokenClothing /cut _tokenClothing2"""

        # Chama a API do OpenRouter
        llm_response = call_openrouter_api(system_prompt, user_content, openrouter_token)
        
        if "error" in llm_response:
            error_msg = f"Erro LLM: {llm_response['error']}"
            return (error_msg, error_msg, error_msg, error_msg, error_msg, error_msg)
        
        # Extrai os dados estruturados
        try:
            lora_air_id = civitai_data['air_id']
            character_name = llm_response.get('character_name', 'unknown_character')
            character_description = llm_response.get('character_description', '')
            s1_outfits = llm_response.get('s1', '')
            s2_outfit = llm_response.get('s2', '')
            s3_outfit = llm_response.get('s3', '')
            
            return (lora_air_id, character_name, character_description, s1_outfits, s2_outfit, s3_outfit)
            
        except Exception as e:
            error_msg = f"Erro ao processar resposta: {str(e)}"
            return (error_msg, error_msg, error_msg, error_msg, error_msg, error_msg)

# Registro do nó no ComfyUI
NODE_CLASS_MAPPINGS = {
    "CivitAIInfoProcessor": CivitAIInfoProcessor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CivitAIInfoProcessor": "Packreator Processor"
}