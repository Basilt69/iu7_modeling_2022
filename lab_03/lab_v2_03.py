import streamlit as st
import streamlit.components.v1 as components
from st_aggrid import AgGrid
from streamlit_agraph import agraph, Node, Edge, Config
from math import fabs
from numpy.linalg.linalg import LinAlgError
import numpy as np
import pandas as pd
import networkx as nx
from pyvis.network import Network

import matplotlib.pyplot as plt
import plotly.graph_objects as go

TIME_DELTA = 1e-3
SEED = 17


def show_tz():
    st.markdown("""
        Есть система S с количеством состояний от 1 до 10 (N). Необходимо задать количество состояний, тут же появляется матрица в которой мы должны указать на пересечении Si с Sj (S-итого с S-житым) интенсивность перехода из состояния в состояние, если она (интенсивность) есть. Строки S1, S2, ..., Sn. И столбец точно также пронумерован. Мы можем переходить из S1 в S1, переход такой возможен.

        Необходимо найти предельные вероятности нахождения системы в том или ином состоянии, т.е. при t стремящемся к бесконечности, и время. Если с вероятностями всё просто, то время - это не вероятность. И это даже совсем не пропорционально вероятности. Нужно найти время, когда эта система попадает в установившееся состояние.

        Лямбду задаёт пользователь (пересечение столбцов и строк матрицы, интенсивность).

        Правая часть уравнения нуль, производной здесь нету, установившийся режим, константа. Решаем уравнение и находим вероятность.

        Время не может оказаться равным вероятности!
    """)


def get_start_probabilities(n, all_equal=True):
    if all_equal:
        return [1 / n] * n
    else:
        res = [0] * n
        res[0] = 1
        return res


def output(title, caption, data, n=2):
    st.write(title)
    for i in range(len(data)):
        st.write(f"{caption}_{i} {round(fabs(data[i]), n)}")


def calc_probas(matrix, n):
    a = np.zeros((n, n))  # матрица для решения СЛАУ
    b = np.zeros(n)  # матрица для результатов

    for i in range(n - 1):
        for j in range(n):
            if i != j:
                a[i][j] += matrix[j][i]
            else:
                a[j][j] -= sum(matrix[j])
    a[-1] = np.ones(n)
    b[-1] = 1  # нормализация матрицы

    try:
        p = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        p = np.zeros(n)

    return p, b


@st.cache()
def get_data(n: int, vals: int) -> pd.DataFrame:
    arr_0 = np.zeros((n, n)).reshape(-1, n)
    arr_1 = np.ones((n, n)).reshape(-1, n)
    cols = [f"S_{i}" for i in range(n)]
    if vals == 0:
        df = pd.DataFrame(arr_0, columns=cols)
    elif vals == 1:
        df = pd.DataFrame(arr_1, columns=cols)
    else:
        # df = pd.DataFrame(np.random.randint(0, 10, size=n*n).reshape(-1, n), columns=cols)
        df = pd.DataFrame(np.random.randint(0, 2, size=n * n).reshape(-1, n), columns=cols)  # TODO fix
    return df


def plot_probability_over_time(probabilities, stabilization_time, times, probabilities_over_time):
    fig, ax = plt.subplots()
    for i_node in range(len(probabilities_over_time[0])):
        ax.plot(times, [i[i_node] for i in probabilities_over_time])
        ax.scatter(stabilization_time[i_node], probabilities[i_node])

    plt.title("Время стабилизации системы")
    ax.legend([f"S_{i}" for i in range(len(probabilities))])
    plt.xlabel('Время (t)')
    plt.ylabel('Вероятность (p)')
    plt.grid(True)
    st.pyplot(fig)

    # # interactive plotly plot (масло масляное
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(
    #     x=np.arange(10),
    #     y=probabilities,
    #     mode='lines',
    # ))
    # fig.update_layout(
    #     title_text="Время стабилизации системы",
    #     xaxis_title="Время(t)",
    #     yaxis_title="Вероятность (p)",
    #     # showlegend=False
    # )
    # st.write(fig)


def plot_graph(graph):
    for i in range(len(graph)):
        graph.nodes[i]["title"] = f"S_{i}"
    nt = Network("600px", "600px", notebook=True, font_color="grey", heading="Граф")
    nt.from_nx(graph)
    # physics = st.checkbox("Добавим немного физики?")
    # if physics:
    #     nt.show_buttons(filter_=["physics"])
    nt.show("Markov_chain.html")

    HtmlFile = open("Markov_chain.html", "r", encoding="utf-8")
    source_code = HtmlFile.read()
    components.html(source_code, height=1200, width=1000)


def plot_graph2(graph):
    nodes = [Node(id=i, label=str(i), size=200) for i in range(len(graph.nodes))]
    edges = [Edge(source=i, target=j, type="CURVE_SMOOTH") for (i, j) in graph.edges]

    config = Config(width=500,
                    height=500,
                    directed=True,
                    nodeHighlightBehavior=True,
                    highlightColor="#F7A7A6",
                    collapsible=True,
                    node={'labelProperty': 'label'},
                    link={'labelProperty': 'label', 'renderLabel': True}
                    )

    return_value = agraph(nodes=nodes,
                          edges=edges,
                          config=config)


def dps(matrix, probabilities):
    n = len(matrix)
    return [
        TIME_DELTA * sum(
            [
                probabilities[j] * (-sum(matrix[i]) + matrix[i][i])
                if i == j else
                probabilities[j] * matrix[j][i]
                for j in range(n)
            ]
        )
        for i in range(n)
    ]


def calc_stabilization_times(matrix, start_probabilities, limit_probabilities, n, current_time=0):
    current_probabilities = start_probabilities.copy()
    stabilization_times = [0 for i in range(n)]
    # stabilization_times = [1 for i in range(n)]  # TODO drop after tests

    # total_lambda_sum = sum([sum(i) for i in matrix]) * SEED
    total_lambda_sum = np.sum(matrix) * SEED
    cool_eps = [p / total_lambda_sum for p in limit_probabilities]

    while not all(stabilization_times):
        curr_dps = dps(matrix, current_probabilities)
        for i in range(n):
            if (not stabilization_times[i] and curr_dps[i] <= cool_eps[i] and
                    abs(current_probabilities[i] - limit_probabilities[i]) <= cool_eps[i]):
                stabilization_times[i] = current_time
            current_probabilities[i] += curr_dps[i]

        current_time += TIME_DELTA

    return stabilization_times


def calc_probability_over_time(matrix, start_probabilities, end_time):
    n = len(matrix)
    current_time = 0
    current_probabilities = start_probabilities.copy()

    probabilities_over_time = []
    times = []

    while current_time < end_time:
        probabilities_over_time.append(current_probabilities.copy())
        curr_dps = dps(matrix, current_probabilities)
        for i in range(n):
            current_probabilities[i] += curr_dps[i]

        current_time += TIME_DELTA

        times.append(current_time)

    return times, probabilities_over_time


def main():
    st.header("Моделирование. Лабораторная работа №2")
    st.write("Предельные вероятности состояний. Уравнения Колмогорова")

    if st.checkbox("Показать ТЗ"):
        show_tz()

    c1, c2 = st.beta_columns(2)
    N = c1.slider("Задайте количество состояний системы (N):", min_value=1, max_value=10, value=5)
    values = c2.selectbox("Заполнить? (единицами, случайно):", (1, "случайными значениями"))
    df = get_data(N, values)

    st.subheader("Введите значения интенсивности переходов (λ):")
    grid_return = AgGrid(
        df,
        editable=True,
        # sortable=False,
        # filter=False,
        # resizable=False,
        # defaultWidth=5,
        # fit_columns_on_grid_load=True,
        reload_data=False,
    )

    arr = grid_return["data"].to_numpy()

    # Находим предельные вероятности
    probas, start_probas = calc_probas(arr, N)
    st.write("Средний процент времени нахождения системы в предельном режиме в состоянии n:")
    for i in range(N):
        pr = round(probas[i], 2)
        perc = round(pr * 100, 2)
        st.write(f"S_{i} - {perc}%")
    output('Предельные вероятности:', 'p', probas)

    # Находим время стабилизации
    start_probabilities = get_start_probabilities(N, all_equal=False)
    stabilization_time = calc_stabilization_times(arr.tolist(), start_probas.tolist(), probas, N)  # TODO fix
    output('Время стабилизации:', 't', stabilization_time)

    # Выводим графики вероятностей как функции времени
    times, probabilities_over_time = calc_probability_over_time(arr, start_probabilities, 5)
    plot_probability_over_time(probas, stabilization_time, times, probabilities_over_time)  # TODO fix

    # Рисуем графы
    G = nx.from_numpy_array(arr, create_using=nx.DiGraph)
    plot_graph(G)
    plot_graph2(G)


if __name__ == "__main__":
    main()