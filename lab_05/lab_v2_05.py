'''
Разработать программу для моделирования следующей системы:

в инфоормационный центр приходят клиенты через интервал времени 10 +-2 минуты. Если все три имеющихся оператора заняты,
клиенту отказывают в обслуживании. Операторы имеют разную производительность и могут обеспечивать обслуживание среднего
запроса пользователя за 20+-5, 40+-10, 40 +-20. Клиенты стремятся занять свободного оператора с максимальной
производительностью. Полученные запросы сдаются в накопитель. Откуда выбираются на обработку. На первый компьютер запросы
от 1 и 2-го операторов, на второй - запросы от 3-его. Время обработки запросов 1-м и 2-м компьютером равны 15 и 30 мин.
соответственно. Необходимо промоделировать процесс обработки 300 запросов.
'''
import streamlit as st
import numpy.random as nr


class UniformGenerator:
    def __init__(self, m, d):
        self._a = m - d
        self._b = m + d
        if not 0 <= self._a <= self._b:
            raise ValueError('Параметры должны удовлетворять условию 0 <= a <= b')

    def next(self):
        return nr.uniform(self._a, self._b)


class ConstGenerator:
    def __init__(self, m):
        if m <= 0:
            raise ValueError('Параметр должен быть больше 0')
        self._m = m

    def next(self):
        return self._m


class RequestGenerator:
    def __init__(self, generator):
        self._generator = generator
        self._receivers = []
        self._generated_requests = 0
        self._next_event_time = 0

    @property
    def next_event_time(self):
        return self._next_event_time

    @next_event_time.setter
    def next_event_time(self, time):
        self._next_event_time = time

    @property
    def generated_requests(self):
        return self._generated_requests

    def add_receiver(self, receiver):
        if receiver not in self._receivers:
            self._receivers.append(receiver)

    def remove_receiver(self, receiver):
        try:
            self._receivers.remove(receiver)
        except KeyError:
            pass

    def generate_time(self):
        return self._generator.next()

    def emit_request(self):
        self._generated_requests += 1
        for receiver in self._receivers:
            if receiver.receive_request():
                return receiver
        else:
            return None


class RequestProcessor(RequestGenerator):
    def __init__(self, generator, *, max_queue_size=0):
        super().__init__(generator)
        self._generator = generator
        self._queued_requests = 0
        self._max_queue_size = max_queue_size
        self._queue_size = max_queue_size
        self._processed_requests = 0

    @property
    def processed_requests(self):
        return self._processed_requests

    @property
    def queue_size(self):
        return self._queue_size

    @queue_size.setter
    def queue_size(self, size):
        self._queue_size = size

    @property
    def queued_requests(self):
        return self._queued_requests

    @property
    def reentered_requests(self):
        return self._reentered_requests

    def process(self):
        if self._queued_requests > 0:
            self._processed_requests += 1
            self._queued_requests -= 1
            self.emit_request()

    def receive_request(self):
        if self._max_queue_size == 0:
            if self._queued_requests >= self._queue_size:
                self._queue_size += 1
            self._queued_requests += 1
            return True
        elif self._queued_requests < self._queue_size:
            self._queued_requests += 1
            return True
        return False


def event_based_modelling(client_m, client_d,
                          op0_m, op0_d, op1_m, op1_d, op2_m, op2_d,
                          comp0_m, comp1_m, c_count):
    client_gen = RequestGenerator(UniformGenerator(client_m, client_d))
    op0 = RequestProcessor(UniformGenerator(op0_m, op0_d), max_queue_size=1)
    op1 = RequestProcessor(UniformGenerator(op1_m, op1_d), max_queue_size=1)
    op2 = RequestProcessor(UniformGenerator(op2_m, op2_d), max_queue_size=1)
    comp0 = RequestProcessor(ConstGenerator(comp0_m))
    comp1 = RequestProcessor(ConstGenerator(comp1_m))

    # добавляются в список в порядке приоритета
    client_gen.add_receiver(op0)
    client_gen.add_receiver(op1)
    client_gen.add_receiver(op2)
    op0.add_receiver(comp0)
    op1.add_receiver(comp0)
    op2.add_receiver(comp1)

    devices = [client_gen, op0, op1, op2, comp0, comp1]

    # начальная инициализация времени
    for device in devices:
        device.next_event_time = 0

    dropped_requests = 0
    client_gen.next_event_time = client_gen.generate_time()
    while client_gen.generated_requests < c_count:
        current_time = client_gen.next_event_time
        # синхронизация времени
        for device in devices:
            if 0 < device.next_event_time < current_time:
                current_time = device.next_event_time

        for device in devices:
            if current_time == device.next_event_time:
                if not isinstance(device, RequestProcessor):
                    assigned_processor = client_gen.emit_request()
                    if assigned_processor is not None:
                        assigned_processor.next_event_time = (current_time +
                                                              assigned_processor.generate_time())
                    else:
                        dropped_requests += 1
                    client_gen.next_event_time = current_time + client_gen.generate_time()
                else:
                    device.process()
                    if device.queued_requests == 0:
                        device.next_event_time = 0

                    else:
                        device.next_event_time = current_time + device.generate_time()

    return round(dropped_requests / c_count, 4), dropped_requests


def show_tz():
    st.markdown("""
        Разработать программу для моделирования следующей системы: в
        информационный центр приходят клиенты через интервал времени 10 +- 2
        минуты. Если все три имеющихся оператора заняты, клиенту отказывают в
        обслуживании. Операторы имеют разную производительность и могут
        обеспечивать обслуживание среднего запроса пользователя за 20 +- 5; 40 +-
        10; 40 +- 20. Клиенты стремятся занять свободного оператора с максимальной
        производительностью. Полученные запросы сдаются в накопитель. Откуда
        выбираются на обработку. На первый компьютер запросы от 1 и 2-ого
        операторов, на второй – запросы от 3-его. Время обработки запросов первым
        и 2-м компьютером равны соответственно 15 и 30 мин. Промоделировать
        процесс обработки 300 запросов.
        """)


def main():
    st.header("Моделирование. Лабораторная работа №4")
    st.write("Многоканальная СМО")

    if st.checkbox("Показать ТЗ"):
        show_tz()
    st.markdown("---")

    st.write("Клиенты")
    c0, c1, c2 = st.columns(3)
    n = c0.number_input("Количество запросов:", min_value=0, max_value=10000, value=300)
    client_m = c1.number_input("Интервал (мин.):", min_value=.01, max_value=10000., value=10.)
    client_d = c2.number_input("±:", min_value=.01, max_value=10000., value=2.)

    st.write("Операторы")
    c3, c4 = st.columns(2)
    op1_m = c3.number_input("№1. Интервал (мин.):", min_value=.01, max_value=10000., value=20.)
    op1_d = c4.number_input(" ±:", min_value=.01, max_value=10000., value=5.)

    c5, c6 = st.columns(2)
    op2_m = c5.number_input("№2. Интервал (мин.):", min_value=.01, max_value=10000., value=40.)
    op2_d = c6.number_input("  ±:", min_value=.01, max_value=10000., value=10.)

    c7, c8 = st.columns(2)
    op3_m = c7.number_input("№3. Интервал (мин.):", min_value=.01, max_value=10000., value=40.)
    op3_d = c8.number_input("   ±:", min_value=.01, max_value=10000., value=20.)

    st.write("Компьютеры")
    c9, c10 = st.columns(2)
    comp1 = c9.number_input("Компьютер №1. Время обработки (мин.):", min_value=0.01, max_value=10000., value=15.)
    comp2 = c10.number_input("Компьютер №2. Время обработки (мин.):", min_value=0.01, max_value=10000., value=30.)

    denial_probability, missed_clients = event_based_modelling(
        client_m, client_d, op1_m, op1_d, op2_m, op2_d, op3_m, op3_d, comp1, comp2, n
    )

    st.button("Моделировать")

    st.write("Результаты моделирования:")
    r1, r2 = st.columns(2)
    r1.code(f"Вероятность отказа: {denial_probability}")
    r2.code(f"Кол-во необработанных заявок: {missed_clients}")


if __name__ == "__main__":
    main()