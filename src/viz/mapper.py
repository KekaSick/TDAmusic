"""
viz/mapper.py
-------------
Mapper (kmapper) для отображения структурных частей песни.
ВАЖНО: Mapper строится на DIRECT-облаке кадров, НЕ на Takens-облаке.
Узел = группа похожих моментов (например, повторяющийся припев).
"""
import kmapper as km
import sklearn
import numpy as np


def build_mapper_graph(cloud_direct, out_file="mapper_graph.html", title="Song Mapper Graph"):
    """
    Строит Mapper-граф.
    cloud_direct: (N, D) - исходные эмбеддинги (до окна Такенса), например после PCA.
    out_file: путь к HTML-файлу для сохранения.
    """
    mapper = km.KeplerMapper(verbose=1)

    # Фильтр: индекс времени, чтобы выявить повторяющиеся секции во времени
    # Mapper разобьет временную шкалу на интервалы, а кластеризация 
    # объединит кадры внутри интервалов по их D-мерному расстоянию.
    # Если мы хотим, чтобы одинаковые части песни из *разного* времени соединялись,
    # мы должны фильтровать так, чтобы похожие куски имели близкие значения фильтра.
    # Но стандартный подход TDA для time-series: фильтр = проекция (PCA) или L2-норма.
    # Сделаем фильтр 2D: (время, PCA_1) или просто (PCA_1, PCA_2).
    # Используем PCA_1 и PCA_2 как фильтр, так как они ловят главные факторы.
    
    # Чтобы было надежно для музыки, возьмем PCA_1 и PCA_2 как линзу.
    lens = mapper.fit_transform(cloud_direct, projection=sklearn.decomposition.PCA(n_components=2))

    # Кластеризация: DBSCAN
    clusterer = sklearn.cluster.DBSCAN(eps=0.5, min_samples=3)

    # Создаем граф
    graph = mapper.map(
        lens,
        cloud_direct,
        cover=km.Cover(n_cubes=10, perc_overlap=0.3),
        clusterer=clusterer
    )

    # Визуализация
    mapper.visualize(
        graph,
        path_html=out_file,
        title=title,
        custom_tooltips=np.arange(len(cloud_direct)) # показываем время в тултипах
    )
    return graph
