import streamlit as st
#from lab_01 import pseudo_random_nums
#from lab_03 import kolmogorov
#from lab_04 import imit
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
        index=0
    )

    '''if lab[:1] == "1":
        pseudo_random_nums.main()
    elif lab[:1] == "2":
        kolmogorov.main()

    elif lab[:1] == "3":
        imit.main()

    elif lab[:1] == "4":
        lab_v2_05.main()'''
    if lab[:1] == "4":
        lab_v2_05.main()



if __name__ == "__main__":
    main()
