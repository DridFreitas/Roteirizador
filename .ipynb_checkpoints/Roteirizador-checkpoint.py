# -*- coding: latin1 -*-
import pandas as pd
import os
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import googlemaps
import time
from geopy.distance import geodesic
from sklearn.cluster import KMeans

# Configurar a conexão com o MySQL
DATABASE_URI = 'mysql+pymysql://root:root@localhost:3306/roteirizador'
engine = create_engine(DATABASE_URI)

# Nomes das tabelas no banco de dados
tabela_lojas = 'lojas'
tabela_tecnicos = 'tecnicos'

# Nomes dos arquivos de dados
arquivo_escala = 'escala.csv'  # Substitua pelo nome real do seu arquivo CSV
planilha_chamados = 'incident.xlsx'  # Substitua pelo nome real do seu arquivo Excel

# Definir o diretório base e os nomes dos arquivos
diretorio_base = os.getcwd()
incident_file = 'INCIDENT.xlsx'
escala_csv = 'escala.csv'

# Substitua 'YOUR_API_KEY' pela sua chave de API do Google Maps
gmaps = googlemaps.Client(key='AIzaSyCqEfEQZWNeWb08cBRPwZ_YMkT4o-J9I2E')

# Função para calcular distância e duração usando a API do Google Maps
def calcular_distancia_duracao(origem, destino):
    """
    Calcula a distância e a duração entre dois pontos geográficos usando a API do Google Maps.
    
    :param origem: Tupla (latitude, longitude) do ponto de origem.
    :param destino: Tupla (latitude, longitude) do ponto de destino.
    :return: Distância em metros e duração em segundos.
    """
    directions_result = gmaps.directions(origem, destino, mode="driving", departure_time="now")
    
    if not directions_result:
        raise ValueError("Não foi possível obter direções entre os pontos fornecidos.")
    
    leg = directions_result[0]['legs'][0]
    distancia = leg['distance']['value']  # Distância em metros
    duracao = leg['duration_in_traffic']['value']  # Duração em segundos, considerando o trânsito atual
    
    return distancia, duracao

# Função principal para geração de roteiro
def gerar_roteiro(arquivo_escala, planilha_chamados):
    start_time = time.time()  # Iniciar a contagem do tempo

    # Carregar a escala de técnicos e definir dia atual para a filtragem
    dia_atual = int(input("Informe o dia para o qual deseja gerar o roteiro (ex: 27): "))
    print(f"Carregando a escala de técnicos para o dia {dia_atual}...")
    tecnicos_disponiveis_lista, escala_df = obter_tecnicos_disponiveis(arquivo_escala, dia_atual)

    # Confirmar a equipe de técnicos para o dia
    tecnicos_disponiveis_lista = confirmar_equipe(tecnicos_disponiveis_lista, escala_df)
    print(f"Técnicos disponíveis carregados: {len(tecnicos_disponiveis_lista)} técnicos.")

    # Carregar dados de técnicos e lojas do banco de dados
    print("Carregando dados de técnicos e lojas do banco de dados...")
    lojas_df = pd.read_sql('SELECT * FROM lojas', con=engine)
    tecnicos_df = pd.read_sql('SELECT * FROM tecnicos', con=engine)

    # Padronizar nomes das colunas para maiúsculas
    lojas_df.columns = lojas_df.columns.str.upper()
    tecnicos_df.columns = tecnicos_df.columns.str.upper()

    # Verificar se as colunas LATITUDE e LONGITUDE existem
    if 'LATITUDE' not in lojas_df.columns or 'LONGITUDE' not in lojas_df.columns:
        raise ValueError("Colunas 'LATITUDE' e 'LONGITUDE' não encontradas em lojas_df")
    if 'LATITUDE' not in tecnicos_df.columns or 'LONGITUDE' not in tecnicos_df.columns:
        raise ValueError("Colunas 'LATITUDE' e 'LONGITUDE' não encontradas em tecnicos_df")

    # Filtrar técnicos disponíveis com base na escala
    tecnicos_disponiveis_df = tecnicos_df[tecnicos_df['NOME'].isin(tecnicos_disponiveis_lista)]

    # Carregar planilha de chamados
    print("Carregando planilha de chamados...")
    chamados_df = pd.read_excel(planilha_chamados, sheet_name=0)
    chamados_df.columns = chamados_df.columns.str.upper()  # Padronizar nomes das colunas para maiúsculas
    chamados_df['CENTRO DE CUSTO'] = chamados_df['CENTRO DE CUSTO'].astype(str).str.zfill(4)  # Padronizar CENTRO DE CUSTO

    # Combinar informações de chamados com a base de lojas
    lojas_df['CENTRO DE CUSTO'] = lojas_df['CENTRO DE CUSTO'].astype(str).str.zfill(4)  # Padronizar CENTRO DE CUSTO para 4 dígitos
    lojas_df = lojas_df.merge(chamados_df, on='CENTRO DE CUSTO', how='inner')

    # Alocar técnicos por proximidade geográfica
    lojas_df = alocar_tecnicos_por_proximidade(lojas_df, tecnicos_disponiveis_df)

    # Verificar se a coluna 'TECNICO' foi adicionada corretamente
    if 'TECNICO' not in lojas_df.columns:
        raise ValueError("Coluna 'TECNICO' não foi adicionada corretamente ao DataFrame de lojas após a alocação.")

    # Imprimir o DataFrame lojas_df para depuração
    print("DataFrame lojas_df após alocação de técnicos:")
    print(lojas_df.head())

    # Otimizar a jornada diária dos técnicos e gerar a planilha de roteiro
    roteiro_data = []
    for _, tecnico in tecnicos_disponiveis_df.iterrows():
        atendimentos = otimizar_jornada(tecnico, lojas_df)
        print(f"Roteiro para o técnico {tecnico['NOME']}:")
        hora_atual = datetime.now()
        for i, atendimento in enumerate(atendimentos.itertuples(), start=1):
            distancia, duracao = calcular_distancia_duracao(
                (tecnico['LATITUDE'], tecnico['LONGITUDE']),
                (atendimento.LATITUDE, atendimento.LONGITUDE)
            )
            hora_atual += timedelta(seconds=duracao)
            roteiro_data.append({
                'numero': i,
                'centro de custo': getattr(atendimento, 'CENTRO DE CUSTO', ''),
                'ordem': getattr(atendimento, 'ORDEM', ''),
                'tag': getattr(atendimento, 'TAG', ''),
                'origem': tecnico['NOME'],
                'destino': atendimento.NOME,
                'km': distancia / 1000,  # Converter metros para quilômetros
                'tempo': duracao / 3600,  # Converter segundos para horas
                'hora do dia prevista': hora_atual.strftime('%H:%M:%S')
            })
            print(roteiro_data[-1])

    # Gerar a planilha de roteiro
    roteiro_df = pd.DataFrame(roteiro_data)
    roteiro_df.to_excel('roteiro_tecnicos.xlsx', index=False)
    print("Planilha de roteiro gerada: roteiro_tecnicos.xlsx")

    end_time = time.time()  # Finalizar a contagem do tempo
    elapsed_time = end_time - start_time  # Calcular o tempo decorrido
    print(f"Tempo total de execução: {elapsed_time:.2f} segundos")

# Função para carregar e verificar a escala de técnicos para o dia informado
def obter_tecnicos_disponiveis(arquivo_escala, dia):
    try:
        escala_df = pd.read_csv(arquivo_escala, delimiter=';', encoding='latin1')
        escala_df.columns = escala_df.columns.str.strip()
        tecnicos_disponiveis_df = escala_df[escala_df[str(dia)] == "."]
        tecnicos_disponiveis_lista = tecnicos_disponiveis_df['NOME'].tolist()
        return tecnicos_disponiveis_lista, escala_df
    except Exception as e:
        print(f"Erro ao carregar escala de técnicos: {e}")
        return [], pd.DataFrame()

# Função para confirmar equipe disponível (inclusão e exclusão de técnicos)
def confirmar_equipe(tecnicos_disponiveis_lista, escala_df):
    equipe_confirmada = False
    while not equipe_confirmada:
        print(f"Técnicos disponíveis para o dia:")
        for i, tecnico in enumerate(tecnicos_disponiveis_lista, start=1):
            print(f"{i}. {tecnico}")

        acao = input("Deseja incluir (I) ou excluir (E) alguém da equipe? Ou pressione ENTER para confirmar: ").upper()

        if acao == 'I':
            tecnicos_folga = [tecnico for tecnico in escala_df['NOME'] if tecnico not in tecnicos_disponiveis_lista]
            if tecnicos_folga:
                print("Técnicos em folga (disponíveis para inclusão):")
                for i, tecnico in enumerate(tecnicos_folga, start=1):
                    print(f"{i}. {tecnico}")
                incluir_index = int(input("Digite o número do técnico que deseja incluir: ")) - 1
                if 0 <= incluir_index < len(tecnicos_folga):
                    tecnicos_disponiveis_lista.append(tecnicos_folga[incluir_index])
                else:
                    print("Número inválido. Tente novamente.")
            else:
                print("Não há técnicos disponíveis para inclusão.")
        elif acao == 'E':
            excluir_index = int(input("Digite o número do técnico que deseja excluir: ")) - 1
            if 0 <= excluir_index < len(tecnicos_disponiveis_lista):
                tecnicos_disponiveis_lista.pop(excluir_index)
            else:
                print("Número inválido. Tente novamente.")
        else:
            equipe_confirmada = True
    return tecnicos_disponiveis_lista

# Função para alocar técnicos por proximidade geográfica
def alocar_tecnicos_por_proximidade(lojas_df, tecnicos_disponiveis_df):
    lojas_df['TECNICO'] = None
    for _, loja in lojas_df.iterrows():
        distancias = tecnicos_disponiveis_df.apply(
            lambda tecnico: geodesic(
                (loja['LATITUDE'], loja['LONGITUDE']),
                (tecnico['LATITUDE'], tecnico['LONGITUDE'])
            ).meters,
            axis=1
        )
        tecnico_mais_proximo = tecnicos_disponiveis_df.loc[distancias.idxmin()]
        lojas_df.at[loja.name, 'TECNICO'] = tecnico_mais_proximo['NOME']
    return lojas_df

# Função para otimizar a jornada diária dos técnicos
def otimizar_jornada(tecnico, lojas_df):
    atendimentos = lojas_df[lojas_df['TECNICO'] == tecnico['NOME']].copy()
    if atendimentos.empty:
        return []

    coordenadas = atendimentos[['LATITUDE', 'LONGITUDE']].values
    kmeans = KMeans(n_clusters=1, random_state=0).fit(coordenadas)
    centroide = kmeans.cluster_centers_[0]

    atendimentos['DISTANCIA_CENTROIDE'] = atendimentos.apply(
        lambda row: geodesic((row['LATITUDE'], row['LONGITUDE']), centroide).meters,
        axis=1
    )
    atendimentos = atendimentos.sort_values(by='DISTANCIA_CENTROIDE')
    return atendimentos

# Executar a função principal
if __name__ == "__main__":
    gerar_roteiro(arquivo_escala, planilha_chamados)

# Função para calcular a matriz de distâncias e durações usando a API do Google Maps
def calcular_matriz_distancia(origens, destinos):
    """
    Calcula a matriz de distâncias e durações entre múltiplos pontos usando a API do Google Maps.
    
    :param origens: Lista de tuplas (latitude, longitude) dos pontos de origem.
    :param destinos: Lista de tuplas (latitude, longitude) dos pontos de destino.
    :return: Matriz de distâncias e durações.
    """
    matrix = gmaps.distance_matrix(origens, destinos, mode="driving", departure_time="now")
    return matrix

# Função para gerar uma imagem estática de um mapa com marcadores usando a API do Google Maps
def gerar_mapa_estatico(centro, marcadores):
    """
    Gera uma imagem estática de um mapa com marcadores usando a API do Google Maps.
    
    :param centro: Tupla (latitude, longitude) do centro do mapa.
    :param marcadores: Lista de tuplas (latitude, longitude) dos marcadores.
    :return: URL da imagem estática do mapa.
    """
    marcador_str = '|'.join([f"{lat},{lng}" for lat, lng in marcadores])
    mapa_url = gmaps.static_map(center=centro, zoom=12, size=(600, 400), markers=marcador_str)
    return mapa_url

# Função para obter rotas otimizadas entre pontos usando a API do Google Maps
def obter_rotas_otimizadas(origem, destino, waypoints):
    """
    Obtém rotas otimizadas entre pontos usando a API do Google Maps.
    
    :param origem: Tupla (latitude, longitude) do ponto de origem.
    :param destino: Tupla (latitude, longitude) do ponto de destino.
    :param waypoints: Lista de tuplas (latitude, longitude) dos pontos intermediários.
    :return: Rotas otimizadas.
    """
    rotas = gmaps.directions(origem, destino, waypoints=waypoints, optimize_waypoints=True, mode="driving", departure_time="now")
    return rotas

# Função para otimizar a sequência de visitas a múltiplos pontos usando a API do Google Maps
def otimizar_sequencia_visitas(pontos):
    """
    Otimiza a sequência de visitas a múltiplos pontos usando a API do Google Maps.
    
    :param pontos: Lista de tuplas (latitude, longitude) dos pontos a serem visitados.
    :return: Sequência otimizada de visitas.
    """
    origem = pontos[0]
    destino = pontos[-1]
    waypoints = pontos[1:-1]
    rotas_otimizadas = obter_rotas_otimizadas(origem, destino, waypoints)
    return rotas_otimizadas

# Exemplo de uso das novas funções no código principal
if __name__ == "__main__":
    # Carregar dados de técnicos e lojas do banco de dados
    lojas_df = pd.read_sql('SELECT * FROM lojas', con=engine)
    tecnicos_df = pd.read_sql('SELECT * FROM tecnicos', con=engine)

    # Padronizar nomes das colunas para maiúsculas
    lojas_df.columns = lojas_df.columns.str.upper()
    tecnicos_df.columns = tecnicos_df.columns.str.upper()

    # Verificar se as colunas LATITUDE e LONGITUDE existem
    if 'LATITUDE' not in lojas_df.columns or 'LONGITUDE' not in lojas_df.columns:
        raise ValueError("Colunas 'LATITUDE' e 'LONGITUDE' não encontradas em lojas_df")
    if 'LATITUDE' not in tecnicos_df.columns or 'LONGITUDE' not in tecnicos_df.columns:
        raise ValueError("Colunas 'LATITUDE' e 'LONGITUDE' não encontradas em tecnicos_df")

    # Filtrar técnicos disponíveis com base na escala
    tecnicos_disponiveis_df = tecnicos_df[tecnicos_df['NOME'].isin(tecnicos_disponiveis_lista)]

    # Carregar planilha de chamados
    chamados_df = pd.read_excel(planilha_chamados, sheet_name=0)
    chamados_df.columns = chamados_df.columns.str.upper()  # Padronizar nomes das colunas para maiúsculas
    chamados_df['CENTRO DE CUSTO'] = chamados_df['CENTRO DE CUSTO'].astype(str).str.zfill(4)  # Padronizar CENTRO DE CUSTO

    # Combinar informações de chamados com a base de lojas
    lojas_df['CENTRO DE CUSTO'] = lojas_df['CENTRO DE CUSTO'].astype(str).str.zfill(4)  # Padronizar CENTRO DE CUSTO para 4 dígitos
    lojas_df = lojas_df.merge(chamados_df, on='CENTRO DE CUSTO', how='inner')

    # Alocar técnicos por proximidade geográfica
    lojas_df = alocar_tecnicos_por_proximidade(lojas_df, tecnicos_disponiveis_df)

    # Verificar se a coluna 'TECNICO' foi adicionada corretamente
    if 'TECNICO' not in lojas_df.columns:
        raise ValueError("Coluna 'TECNICO' não foi adicionada corretamente ao DataFrame de lojas após a alocação.")

    # Imprimir o DataFrame lojas_df para depuração
    print("DataFrame lojas_df após alocação de técnicos:")
    print(lojas_df.head())

    # Otimizar a jornada diária dos técnicos e gerar a planilha de roteiro
    roteiro_data = []
    for _, tecnico in tecnicos_disponiveis_df.iterrows():
        atendimentos = otimizar_jornada(tecnico, lojas_df)
        print(f"Roteiro para o técnico {tecnico['NOME']}:")
        hora_atual = datetime.now()
        for i, atendimento in enumerate(atendimentos.itertuples(), start=1):
            distancia, duracao = calcular_distancia_duracao(
                (tecnico['LATITUDE'], tecnico['LONGITUDE']),
                (atendimento.LATITUDE, atendimento.LONGITUDE)
            )
            hora_atual += timedelta(seconds=duracao)
            roteiro_data.append({
                'numero': i,
                'centro de custo': atendimento.CENTRO_DE_CUSTO,
                'ordem': getattr(atendimento, 'ORDEM', ''),
                'tag': getattr(atendimento, 'TAG', ''),
                'origem': tecnico['NOME'],
                'destino': atendimento.NOME,
                'km': distancia / 1000,  # Converter metros para quilômetros
                'tempo': duracao / 3600,  # Converter segundos para horas
                'hora do dia prevista': hora_atual.strftime('%H:%M:%S')
            })
            print(roteiro_data[-1])

    # Gerar a planilha de roteiro
    roteiro_df = pd.DataFrame(roteiro_data)
    roteiro_df.to_excel('roteiro_tecnicos.xlsx', index=False)
    print("Planilha de roteiro gerada: roteiro_tecnicos.xlsx")

    # Exemplo de uso da Distance Matrix API no código principal
    origens = [(tecnico['LATITUDE'], tecnico['LONGITUDE']) for _, tecnico in tecnicos_disponiveis_df.iterrows()]
    destinos = [(loja['LATITUDE'], loja['LONGITUDE']) for _, loja in lojas_df.iterrows()]
    matriz_distancia = calcular_matriz_distancia(origens, destinos)
    print("Matriz de distância e duração calculada.")

    # Exemplo de uso da Maps Static API para gerar um mapa estático
    centro = (lojas_df['LATITUDE'].mean(), lojas_df['LONGITUDE'].mean())
    marcadores = [(loja['LATITUDE'], loja['LONGITUDE']) for _, loja in lojas_df.iterrows()]
    mapa_url = gerar_mapa_estatico(centro, marcadores)
    print(f"URL do mapa estático: {mapa_url}")

    # Exemplo de uso da Route Optimization API para otimizar a sequência de visitas
    pontos = [(loja['LATITUDE'], loja['LONGITUDE']) for _, loja in lojas_df.iterrows()]
    sequencia_otimizada = otimizar_sequencia_visitas(pontos)
    print(f"Sequência otimizada de visitas: {sequencia_otimizada}")
