'''
Задание: Разработать программу, на на экране которой должна присутствовать таблица, содержащая две большие колонки.
Эти колонки должны отображать табличный и алгоритмический способ получения последовательности псевдослучайных чисел.
Табличные данные, которые необходимо брать из готовых таблиц, содержащих последовательности псевдослучайных чисел, должны
содержать три столбца: одноразрядные, двухразрядные и трёхразрядные. Данные, полученные алгоритмическим способом(генерируем
любым алгоритмом на выбор), должны содержать три столбца: одноразрядные, двухразрядные и трехразряядные. На экран выводим 10-ть
числе из тысячи, которую сгенерируем. Внизу под каждым столбцом выводим число, которое оценивает случайность данной
последовательности. Необходимо придумать критерий случайности. Необходимо предусмотреть форму, в которой пользователь
может сам задать последовательность чисел - после чего выводиться тоговое число с оценкой случайности.

'''

import streamlit as st
import pandas as pd
import numpy as np
import math
import sys

from scipy.stats import norm

# IBMQ
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute, IBMQ, Aer
from qiskit.tools.monitor import job_monitor
from qiskit_rng import Generator

# Lines 10580–10594, columns 21–40, from A Million Random Digits with 100,000 Normal Deviates
RAND_TABLE = """
    73735 45963 78134 63873
    02965 58303 90708 20025
    98859 23851 27965 62394
    33666 62570 64775 78428
    81666 26440 20422 05720

    15838 47174 76866 14330
    89793 34378 08730 56522
    78155 22466 81978 57323
    16381 66207 11698 99314
    75002 80827 53867 37797

    99982 27601 62686 44711
    84543 87442 50033 14021
    77757 54043 46176 42391
    80871 32792 87989 72248
    30500 28220 12444 71840
"""


def ibmq_qrng(num_q, minimum, maximum):
    """
    Generate real random numbers from IBM Quantum computer via API
    :param num_q:
    :param minimum:
    :param maximum:
    :return:
    """
    simulator = Aer.get_backend('qasm_simulator')
    q = QuantumRegister(num_q, 'q')
    c = ClassicalRegister(num_q, 'c')

    circuit = QuantumCircuit(q, c)
    circuit.h(q)  # Applies hadamard gate to all qubits
    circuit.measure(q, c)  # Measures all qubits

    job = execute(circuit, simulator, shots=1)
    counts = job.result().get_counts()
    result = int(counts.most_frequent(), 2)
    result1 = minimum + result % (maximum + 1 - minimum)
    return result1


def user_input_handler(digit_capacity: int, key: int) -> np.array:
    """
    User input handler
    :param digit_capacity: admissible digit capacity
    :param key: button for clear input
    :return:
    """
    placeholder = st.empty()
    text_input = f"Введите 10 чисел с разрядом {digit_capacity} через пробел:"
    user_input = placeholder.text_input(text_input)
    click_clear = st.button('Очистить', key=key)
    if click_clear:
        user_input = placeholder.text_input(text_input, value='', key=key)
    list_nums = user_input.split()
    for num in list_nums:
        if len(num) != digit_capacity:
            st.error(f"Допустимы числа только с разрядом {digit_capacity}. Повторите ввод.")
        try:
            int(num)
        except ValueError as error:
            st.error(f"Вы ввели {num}. Нужно вводить числа. Повторите ввод.")
            return np.zeros(0)

    if list_nums and len(list_nums) != 10:
        st.warning(f"Вы ввели {len(list_nums)}, а нужно 10 чисел. Повторите ввод.")

    return np.array(list_nums, dtype=np.int16)


def fourier_estimator(data: np.ndarray) -> float:
    """
    Discrete Fourier transform (spectral) estimator as described in NIST paper:
    https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-22r1a.pdf
    The focus of this test is the peak heights in the Discrete Fourier Transform of the sequence.
    The purpose of this estimator is to detect periodic features (i.e., repetitive patterns that are near each other) in the
    tested sequence that would indicate a deviation from the assumption of randomness.
    The intention is to detect whether the number of peaks exceeding the 95% threshold is significantly different than 5%.
    The significance value of the estimator is 0.01.
    :param data:
    :return:
    """
    # Convert all the zeros in the array to -1
    data[data == 0] = -1
    # Compute DFT
    discrete_fourier_transform = np.fft.fft(data)
    # Compute magnitudes of first half of sequence depending on the system type
    if sys.version_info > (3, 0):
        magnitudes = abs(discrete_fourier_transform)[:data.size // 2]
    else:
        magnitudes = abs(discrete_fourier_transform)[:data.size / 2]
    # Compute upper threshold
    threshold: float = math.sqrt(math.log(1.0 / 0.05) * data.size)
    # Compute the expected number of peaks (N0)
    expected_peaks: float = 0.95 * data.size / 2.0
    # Count the peaks above the upper threshold (N1)
    counted_peaks: float = float(len(magnitudes[magnitudes < threshold]))
    # Compute the score (P-value) using the normalized difference
    normalized_difference: float = (counted_peaks - expected_peaks) / math.sqrt((data.size * 0.95 * 0.05) / 4)
    score = math.erfc(abs(normalized_difference) / math.sqrt(2))

    significance_value = 0.01
    return False if score >= significance_value else True
    # return (True, score) if score >= significance_value else (False, score)
    # return score


def monotonic_estimator(data: np.ndarray) -> float:
    """
    Monotonic estimator
    :param data:
    :return:
    """
    n = len(data) - 1
    h = 0
    l = 0

    for i in range(1, len(data)):
        if (data[i] - data[i - 1]) > 0:
            h += 1
        else:
            l += 1

    h /= n
    l /= n

    return round(abs(h - l), 2)


def frequency_estimator(data: np.ndarray) -> float:
    """
    Frequency estimator
    :param data:
    :return:
    """
    m = 1 / 2  # мат ожидание массива (идеальный случай)

    l = 0
    r = 0

    for i in data:
        if i < m:
            l += 1
        else:
            r += 1

    l /= len(data)
    r /= len(data)

    # насколько слева чисел больше чем справа
    return round(abs(l - r))


def generate_table(data, columns):
    """
    Generate pandas DataFrame with random samples
    :param data:
    :param columns:
    :return:
    """
    discharges = ["1 разр.", "2 разр.", "3 разр."]
    df = pd.DataFrame(
        data=data.T,
        index=range(1, 11),
        columns=pd.MultiIndex.from_product([columns, discharges])
    )

    df_est = pd.DataFrame(
        data=[
            [round(norm.fit(col)[1], 2) for col in data],
            [norm.fit(col)[0] for col in data],
            [monotonic_estimator(col) for col in data],
            [fourier_estimator(col) for col in data],
        ],
        index=[
            "Стандартное откл.",
            "Среднее откл.",
            "Монотонность",
            "Фурье",
        ],
        columns=["" for i in range(len(df.columns.tolist()))]
    )

    st.dataframe(data=df)
    st.dataframe(data=df_est)


def gen_rnd_smpl(low: int, high: int, size: int = 1000, d_type=np.int16) -> np.array:
    """
    Generate sample of random integers
    :param d_type: desired dtype of the result.
    :param low: min value in array
    :param high: max value in array
    :param size: len of array
    :return: generated sample of random numbers
    """
    return np.random.randint(low, high, size, d_type)[:10]


def main():
    st.header("Моделирование. Лабораторная работа №2")
    st.write("Исследование последовательности псевдослучайных чисел")

    random_type = st.radio(
        "Выберите метод получения чисел",
        ("Алгоритмическая генерация", "Пользовательский ввод", "Пуассон", "Квантовая генерация")
    )

    if random_type == "Алгоритмическая генерация":
        st.markdown("---")
        random_table = RAND_TABLE.replace(" ", "").replace("\n", "")
        table_cap_1 = random_table[:10]
        table_cap_2 = random_table[11:31]
        table_cap_3 = random_table[32:62]

        tbl_1 = np.array([int(s) for s in table_cap_1], dtype=np.int16)
        tbl_2 = np.array([table_cap_2[i:i + 2] for i in range(0, len(table_cap_2), 2)], dtype=np.int16)
        tbl_3 = np.array([table_cap_3[i:i + 3] for i in range(0, len(table_cap_3), 3)], dtype=np.int16)
        alg_1 = gen_rnd_smpl(0, 9)
        alg_2 = gen_rnd_smpl(10, 99)
        alg_3 = gen_rnd_smpl(100, 999)

        st.write("Табличный и алгоритмический метод получения псевдослучайных чисел.")
        generate_table(
            data=np.array([
                tbl_1,
                tbl_2,
                tbl_3,
                alg_1,
                alg_2,
                alg_3,
            ]),
            columns=["Табл.", "Алг."]
        )
        st.button("Сгенерировать")

        if st.checkbox("Показать графики"):
            st.line_chart(tbl_1)
            st.line_chart(tbl_2)
            st.line_chart(tbl_3)
            st.line_chart(alg_1)
            st.line_chart(alg_2)
            st.line_chart(alg_3)

    elif random_type == "Пользовательский ввод":
        st.markdown("---")
        st.write("Пользовательский ввод случайных чисел.")
        user_nums_cap_1 = user_input_handler(digit_capacity=1, key=1)
        user_nums_cap_2 = user_input_handler(digit_capacity=2, key=2)
        user_nums_cap_3 = user_input_handler(digit_capacity=3, key=3)

        if user_nums_cap_1.shape[0] + user_nums_cap_2.shape[0] + user_nums_cap_3.shape[0] == 30:
            generate_table(
                data=np.array([user_nums_cap_1, user_nums_cap_2, user_nums_cap_3]),
                columns=["Польз."]
            )
        if st.checkbox("Показать графики"):
            st.line_chart(user_nums_cap_1)
            st.line_chart(user_nums_cap_2)
            st.line_chart(user_nums_cap_3)

    elif random_type == "Пуассон":
        st.markdown("---")
        st.write("Генерация случайных чисел с использованием распределения Пуассона")
        poisson_cap_1 = np.random.poisson(3, 10)
        poisson_cap_2 = np.random.poisson(33, 10)
        poisson_cap_3 = np.random.poisson(333, 10)

        generate_table(
            data=np.array([poisson_cap_1, poisson_cap_2, poisson_cap_3]),
            columns=["Пуассон"]
        )
        st.button("Сгенерировать")

    elif random_type == "Квантовая генерация":
        st.markdown("---")
        st.write("Генерация случайных чисел с использованием квантового компьютера IBM.")

        api_key = None
        try:
            IBMQ.load_account()
        except Exception as e:
            api_key = st.text_input("Введите IBMQ API Key:")
            if not api_key:
                IBMQ.save_account(api_key, overwrite=True)
                IBMQ.load_account()
        rng_provider = IBMQ.get_provider(hub='ibm-q')
        device = st.selectbox("Select Quantum Device", [
            str(each) for each in rng_provider.backends()
        ])
        backend = rng_provider.get_backend(device)

        generator = Generator(backend=backend)

        if device == "ibmq_qasm_simulator":
            num_q = 32
        else:
            num_q = 5

        quantum_nums_cap_1 = []
        quantum_nums_cap_2 = []
        quantum_nums_cap_3 = []
        for i in range(10):
            quantum_nums_cap_1.append(ibmq_qrng(num_q, 0, 9))
            quantum_nums_cap_2.append(ibmq_qrng(num_q, 10, 99))
            quantum_nums_cap_3.append(ibmq_qrng(num_q, 100, 999))

        generate_table(
            data=np.array([quantum_nums_cap_1, quantum_nums_cap_2, quantum_nums_cap_3]),
            columns=["Квант."]
        )
        st.button("Сгенерировать")
        if st.checkbox("Показать графики"):
            st.line_chart(quantum_nums_cap_1)
            st.line_chart(quantum_nums_cap_2)
            st.line_chart(quantum_nums_cap_3)


if __name__ == "__main__":
    main()