# Image Quality Assurance

Este é um projeto escrito em Python que utiliza o ambiente virtual pipenv para classificar imagens, detectar textos em QR codes e editar imagens com base nos textos lidos.

## Objetivos
1. Classificar imagens como boas ou ruins de acordo com as seguintes tags: luz, foco, vibração, ring light, led e etiqueta.
2. Detectar textos em QR codes presentes nas imagens.
3. Editar as imagens com os textos lidos a partir dos QR codes e disponibilizá-las para download.

## Pré-requisitos
- Python 3.6 ou superior

- pipenv instalado (Caso não tenha, instale através do comando: ```pip install pipenv```)

## Instalação
- Clone o repositório

- Na pasta do projeto, execute o comando: ```pipenv install``` para criar o ambiente virtual e instalar as dependências do projeto.

## Como executar o projeto
- Abra o terminal na pasta do projeto e execute o comando: ```pipenv shell``` para ativar o ambiente virtual.

- Execute o arquivo main.py com o comando: ```streamlit run main.py```

- Faça o upload da imagem jo formato .JPG no campo especificado.
- O programa irá processá-la e exibir a classificação (boa ou ruim) de acordo com os parâmetros e os textos lidos do QR code.

- Caso a imagem seja classificada como boa e contenha um QR code, a imagem será editada com os textos lidos e será disponibilizá-da para download.

Link do projeto na nuvem: https://gleycesouza-image-classification-main-ca86b4.streamlit.app/