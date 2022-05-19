import streamlit as st
from lab_02 import lab_v2_02
from lab_03 import lab_v2_03
from lab_04 import lab_v2_04
from lab_05 import lab_v2_05

st.set_page_config(initial_sidebar_state="collapsed")
st.sidebar.image('logo.png', width=300)


def header():
    author = """
        made by [Василий Ткаченко](https://github.com/Basilt69) 
        for Modelling [labs](https://github.com/Basilt69/iu7_modeling_2022.git)
        in [BMSTU](https://bmstu.ru)
    """
    st.title("МГТУ им. Баумана. Кафедра ИУ7")
    st.write("Преподаватель: Рудаков И.В.")
    st.write("Студент: Ткаченко В.М.")
    st.sidebar.markdown(author)


def main():
    header()
    lab = st.sidebar.radio(
        "Выберите Лабораторную работу", (
            "1. Исследование последовательности псевдослучайных чисел",
            "2. Предельные вероятности состояний. Уравнения Колмогорова",
            "3. Программная имитация i-го прибора",
            "4. МФЦ",
        ),
        index=3
    )

    if lab[:1] == "1":
        lab_v2_02.main()
    elif lab[:1] == "2":
        lab_v2_03.main()
    elif lab[:1] == "3":
        lab_v2_04.main()
    elif lab[:1] == "4":
        lab_v2_05.main()



if __name__ == "__main__":
    main()
