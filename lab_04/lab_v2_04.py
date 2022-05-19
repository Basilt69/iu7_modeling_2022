'''
Разработать программу иммитации i-го прибора. У нас есть генератор (или источник сообщений),есть память и есть
обсулживающий аппарат(ОА). Генератор выдает сообщения по равномерному закону распределенного в интервале a+-b (от a до b).
ОА выбирает сообщения из памяти, и по закону из 1-ой лабы(нормальный). Закон параметрически настраивается. Необходимо
определить минимальную длину очереи (или объем памяти), при котором сообщения не теряются(т.е. не возникает такая
ситуация, когда сообщение идет в ОА, а он занят). Реализовать это надо двумя способами: дельта t и событийно. Посмотреть
есть ли разница. А дальше перелаются выданные сообщения из ОА в процентном соотношении на вход очереди. Например, задается,
что половина сообщений снова поступает на ОА, или 0,7 или 0,1 и снова проверяется, что произойдет с очередью. Определяем
оптимальную длину очереди, т.е. ту длину, при которой ни односообщение необрботанным не исчезает. Т.е. нет отказа(т.е.
необходимо сделать программу, которая сначала выдает минимальное t, потом подает % и она снова выдает минимальное t).
Дополнительно необходимо добавить статистику каждого устройства(мини - GPSS).
'''

import streamlit as st
import scipy.stats as sts
import numpy.random as nr
import numpy as np
import pandas as pd
import random
import math


class UniformDistribution:
    def __init__(self, a, b):
        if not 0 <= a <= b:
            raise ValueError('Параметры должны удовлетворять условию 0 <= a <= b')
        self._a = a
        self._b = b

    def generate(self):
        return nr.uniform(self._a, self._b)


class NormalDistribution:
    def __init__(self, m, d):
        self._m = m
        self._d = d

    def generate(self):
        return nr.normal(self._m, self._d)


class PoissonDistribution:
    def __init__(self, m):
        self._m = m
        self.poisson_rv = sts.poisson(m)

    def _poisson_pmf(self, x):
        if x < 0:
            return 0
        return self.poisson_rv.pmf(x)

    def _poisson_cdf(self, x):
        if x < 0:
            return 0
        return self.poisson_rv.cdf(x)

    def generate(self):
        return self._poisson_cdf(random.random())


class ErlangDistribution:
    def __init__(self, k, l_):
        self._l = l_
        self._k = k

    def _erlang_pdf(self, x):
        if x < 0:
            return 0
        return pow(self._l, self._k + 1) * pow(x, self._k) * np.exp(- self._l * x) / math.factorial(self._k)

    def _erlang_cdf(self, x):
        if x < 0:
            return 0
        return 1 - (1 + self._l * x) * np.exp(-self._l * x)

    def generate(self):
        # return self._erlang_cdf(random.random())
        return nr.gamma(self._k, self._l)


class Model:
    def __init__(self, dt, req_count, reenter_prob):
        self.dt = dt
        self.req_count = req_count
        self.reenter_prob = reenter_prob

        self.queue = 0
        self.queue_len_max = 0
        self.reenter = 0

    def check_len_max(self):
        if self.queue > self.queue_len_max:
            self.queue_len_max = self.queue

    def add_to_queue(self):
        self.queue += 1
        self.check_len_max()

    def rem_from_queue(self, is_reenter=True):
        if self.queue == 0:
            return 0

        self.queue -= 1

        if is_reenter and nr.sample() < self.reenter_prob:
            self.reenter += 1
            self.add_to_queue()

        return 1

    def event_based_modelling(self, a, b, m, d):
        req_generator = UniformDistribution(a, b)
        req_proccessor = NormalDistribution(m, d)

        req_done_count = 0
        t_generation = req_generator.generate()
        t_proccessor = t_generation + req_proccessor.generate()

        while req_done_count < self.req_count:
            if t_generation <= t_proccessor:
                self.add_to_queue()
                t_generation += req_generator.generate()
                continue
            if t_generation >= t_proccessor:
                req_done_count += self.rem_from_queue(True)
                t_proccessor += req_proccessor.generate()

        return self.queue_len_max, self.req_count, self.reenter, round(t_proccessor, 3)

    def time_based_modelling(self, a, b, m, d):

        req_generator = UniformDistribution(a, b)
        req_proccessor = NormalDistribution(m, d)

        req_done_count = 0
        t_generation = req_generator.generate()
        t_proccessor = t_generation + req_proccessor.generate()

        t_curr = 0
        while req_done_count < self.req_count:
            if t_generation <= t_curr:
                self.add_to_queue()
                t_generation += req_generator.generate()
            if t_curr >= t_proccessor:
                if self.queue > 0:
                    req_done_count += self.rem_from_queue(True)
                    t_proccessor += req_proccessor.generate()
                else:
                    t_proccessor = t_generation + req_proccessor.generate()

            t_curr += self.dt

        return self.queue_len_max, self.req_count, self.reenter, round(t_curr, 3)


class Generator:
    def __init__(self, generator):
        self._generator = generator
        self._receivers = set()

    def add_receiver(self, receiver):
        self._receivers.add(receiver)

    def remove_receiver(self, receiver):
        try:
            self._receivers.remove(receiver)
        except KeyError:
            pass

    def next_time(self):
        return self._generator.generate()

    def emit_request(self):
        for receiver in self._receivers:
            receiver.receive_request()


class Processor(Generator):
    def __init__(self, generator, reenter_probability=0):
        super().__init__(generator)
        self.current_queue_size = 0
        self.max_queue_size = 0
        self.processed_requests = 0
        self.reenter_probability = reenter_probability
        self.reentered_requests = 0

    # обрабатываем запрос, если они есть
    def process(self):
        if self.current_queue_size > 0:
            self.processed_requests += 1
            self.current_queue_size -= 1
            self.emit_request()
            # Возвращаем реквест, если срабатывает возвращаемость
            if nr.random_sample() <= self.reenter_probability:
                self.reentered_requests += 1
                # Sself.processed_requests -= 1
                self.receive_request()

    # добавляем реквест в очередь
    def receive_request(self):
        self.current_queue_size += 1
        if self.current_queue_size > self.max_queue_size:
            self.max_queue_size = self.current_queue_size


class Modeller:
    def __init__(self, uniform_a, uniform_b, er_k, er_l, reenter_prop):
        self._generator = Generator(UniformDistribution(uniform_a, uniform_b))
        self._processor = Processor(ErlangDistribution(er_k, er_l), reenter_prop)
        self._generator.add_receiver(self._processor)

    def event_based_modelling(self, request_count):
        generator = self._generator
        processor = self._processor

        gen_period = generator.next_time()
        proc_period = gen_period + processor.next_time()

        while processor.processed_requests < request_count:
            # print( processor.next_time(), generator.next_time())
            if gen_period <= proc_period:
                # появился новый запрос
                # добавляем оправляем его в процессор
                generator.emit_request()
                gen_period += generator.next_time()
            if gen_period >= proc_period:
                # закончилась обработка
                # обрабатываем запрос
                processor.process()

                # проверка для самого первого запроса
                if processor.current_queue_size > 0:
                    proc_period += processor.next_time()
                else:
                    proc_period = gen_period + processor.next_time()

        return (processor.processed_requests, processor.reentered_requests,
                processor.max_queue_size, round(proc_period, 3))

    def time_based_modelling(self, request_count, dt):
        generator = self._generator
        processor = self._processor

        gen_period = generator.next_time()
        proc_period = gen_period
        current_time = 0
        while processor.processed_requests < request_count:
            if gen_period <= current_time:
                # появился новый запрос
                # добавляем оправляем его в процессор
                generator.emit_request()
                gen_period += generator.next_time()
            if current_time >= proc_period:
                # закончилась обработка
                # обрабатываем запрос
                processor.process()
                if processor.current_queue_size > 0:
                    proc_period += processor.next_time()
                else:
                    proc_period = gen_period + processor.next_time()

            # прибавляем дельту
            current_time += dt

        return (processor.processed_requests, processor.reentered_requests,
                processor.max_queue_size, round(current_time, 3))


class ModellerP:
    def __init__(self, uniform_a, uniform_b, mu, reenter_prop):
        self._generator = Generator(UniformDistribution(uniform_a, uniform_b))
        self._processor = Processor(PoissonDistribution(mu), reenter_prop)
        self._generator.add_receiver(self._processor)

    def event_based_modelling(self, request_count):
        generator = self._generator
        processor = self._processor

        gen_period = generator.next_time()
        proc_period = gen_period + processor.next_time()

        while processor.processed_requests < request_count:
            # print( processor.next_time(), generator.next_time())
            if gen_period <= proc_period:
                # появился новый запрос
                # добавляем оправляем его в процессор
                generator.emit_request()
                gen_period += generator.next_time()
            if gen_period >= proc_period:
                # закончилась обработка
                # обрабатываем запрос
                processor.process()

                # проверка для самого первого запроса
                if processor.current_queue_size > 0:
                    proc_period += processor.next_time()
                else:
                    proc_period = gen_period + processor.next_time()

        return (processor.processed_requests, processor.reentered_requests,
                processor.max_queue_size, round(proc_period, 3))

    def time_based_modelling(self, request_count, dt):
        generator = self._generator
        processor = self._processor

        gen_period = generator.next_time()
        proc_period = gen_period
        current_time = 0
        while processor.processed_requests < request_count:
            if gen_period <= current_time:
                # появился новый запрос
                # добавляем оправляем его в процессор
                generator.emit_request()
                gen_period += generator.next_time()
            if current_time >= proc_period:
                # закончилась обработка
                # обрабатываем запрос
                processor.process()
                if processor.current_queue_size > 0:
                    proc_period += processor.next_time()
                else:
                    proc_period = gen_period + processor.next_time()

            # прибавляем дельту
            current_time += dt

        return (processor.processed_requests, processor.reentered_requests,
                processor.max_queue_size, round(current_time, 3))


def show_tz():
    st.markdown("""
        У нас есть генератор (или источник сообщений), есть память, и есть обслуживающий аппарат (ОА). 
        Генератор выдаёт сообщения по равномерному закону распределённого в интервале a+-b (от a до b). 
        ОА выбирает сообщения из памяти, и вот здесь преподаватель пишет наш закон, тот который мы писали во 2-й ЛР 
        (по вариантам, нормальный, экспоненциальный, пуассоновский и т.д.). Все эти законы параметрически настраиваются.
        1. Необходимо определить минимальную длину очереди (или объём памяти), при котором сообщения не теряются 
        (т.е. не возникает такая ситуация, когда сообщение идёт в ОА, а он занят). 
        Реализовывать это нужно двумя способами (принципами): дельта t  и событийно. Посмотреть есть ли разница.

        2. А дальше наступает ужас-ужас. Преподаватель умудряется передать выданные сообщения из ОА в процентном 
        соотношении снова на вход очереди. Он задаёт, например, что половина сообщение снова поступает на ОА, 
        или 0,7 или 0,1 и снова смотрит, что произойдёт с очередью.

        Определить оптимальную длину очереди, т.е. ту длину, при которой ни одно сообщение необработанным не исчезает. 
        Т.е. нет отказа.

        Т.е. нужно сделать прогу, которая сначала выдаст минимальное t, потом подаём % и она снова выдаёт минимальное t.
        И хорошо если ещё и статистики сюда прикрутим и можно будет посмотреть по каждому устройству как оно нагружено, 
        т.е. вроде своего мини-GPSS создадим.
        """)


def main():
    st.header("Моделирование. Лабораторная работа №3")
    st.write("Программная имитация i-го прибора")

    if st.checkbox("Показать ТЗ"):
        show_tz()
    st.markdown("---")

    distribution = st.radio(
        "Выберите тип распределения ОА", (
            "1. Нормальное",
            "2. Пуассоновское",
            "3. Эрланговское",
        )
    )

    st.write("Параметры генератора. Равномерное распределение")
    c1, c2 = st.columns(2)
    a = c1.number_input("Начало интервала (a):", min_value=0, max_value=10000, value=0)
    b = c2.number_input("Конец интервала (b):", min_value=0, max_value=10000, value=10)

    st.write("Дополнительные параметры")
    c4, c5, c6 = st.columns(3)
    requests_count = c4.number_input("Количество запросов:", min_value=1, max_value=10000, value=1000)
    reenter_probability = c5.number_input("Вер-ть повторной обр-ки:", min_value=0., max_value=1., value=.5)
    delta_t = c6.number_input("Значение (Δt):", min_value=0., max_value=1., value=.1)

    if distribution[:1] == "1":
        st.write(f"Параметры обслуживающего аппарата. {distribution[3:]} распределение")
        c2, c3 = st.columns(2)
        m = c2.number_input("Мат. ожидание (μ):", min_value=1., max_value=10000., value=5.)
        d = c3.number_input("Средн.кв. отклонение (σ):", min_value=1., max_value=10000., value=5.)

        modelT = Model(delta_t, requests_count, reenter_probability)
        results1 = modelT.time_based_modelling(a, b, m, d)
        queue_len_max1, req_done_count1, reenter1, time1 = results1

        modelEvent = Model(delta_t, requests_count, reenter_probability)
        results2 = modelEvent.event_based_modelling(a, b, m, d)
        queue_len_max2, req_done_count2, reenter2, time2 = results2

        df = pd.DataFrame({
            "Метод": ["Событийный", "Δt"],
            "Обработанные запросы": [req_done_count2, req_done_count1],
            "Возвращенные запросы": [reenter2, reenter1],
            "Мах длина очереди": [queue_len_max2, queue_len_max1],
            "Время работы": [time2, time1]
        }).T
        st.write(df)

    if distribution[:1] == "2":
        st.write(f"Параметры обслуживающего аппарата. {distribution[3:]} распределение")
        c2, c3 = st.columns(2)
        mu_ = c3.number_input("Мат. ожидание (μ):", min_value=1., max_value=10000., value=5.)

        model = ModellerP(a, b, mu_, reenter_probability)
        result1 = model.event_based_modelling(requests_count)

        model2 = ModellerP(a, b, mu_, reenter_probability)
        result2 = model2.time_based_modelling(requests_count, delta_t)

        df = pd.DataFrame({
            "Метод": ["Событийный", "Δt"],
            "Обработанные запросы": [result1[0], result2[0]],
            "Возвращенные запросы": [result1[1], result2[1]],
            "Мах длина очереди": [result1[2], result2[2]],
            "Время работы": [result1[3], result2[3]]
        }).T
        st.write(df)

    if distribution[:1] == "3":
        st.write(f"Параметры обслуживающего аппарата. {distribution[3:]} распределение")
        c2, c3 = st.columns(2)
        er_k = c2.number_input("Shape (k):", min_value=1, max_value=100, value=1)
        er_lambda = c3.number_input("Scale (λ):", min_value=1, max_value=100, value=2)

        model = Modeller(a, b, er_k, er_lambda, reenter_probability)
        result1 = model.event_based_modelling(requests_count)

        model2 = Modeller(a, b, er_k, er_lambda, reenter_probability)
        result2 = model2.time_based_modelling(requests_count, delta_t)

        df = pd.DataFrame({
            "Метод": ["Событийный", "Δt"],
            "Обработанные запросы": [result1[0], result2[0]],
            "Возвращенные запросы": [result1[1], result2[1]],
            "Мах длина очереди": [result1[2], result2[2]],
            "Время работы": [result1[3], result2[3]]
        }).T
        st.write(df)


if __name__ == "__main__":
    main()