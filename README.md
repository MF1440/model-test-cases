# Моделирование КС (общее) #
Тестовое задание для выполнения в python, либо MATLAB или совместимых средах.

## Требования к выполнению ##
- Ответы на вопросы тестового задания должны получаться без дополнительных манипуляций с кодом, т.е. вызовом отдельного скрипта / функции, проверяющему должно быть очевидно, что нужно сделать для проверки выполнения. Допускается для этого написать отдельную мини-инструкцию, описывающую процесс вызова решения;
- Написание текстового отчета по алгоритму решения не требуется. Хороший код документирует сам себя - крайне рекомендуется использовать понятные названия переменных и комментарии, поясняющие основные шаги при решении (документ с требованиями и рекомендациями, приложенный к заданию, в точности соблюдать не требуется, но он может помочь для структуризации и улучшения качества кода);
- Ожидаемая форма предоставления результата - pull request в исходный репозиторий; при отсутствии достаточного опыта работы с git - допускаются другие, удобные для исполнителя формы;
- Из блока "Задание" требуется выполнить пункт 1, а также один из пунктов 2-9.

## Содержание репозитория ##
Для выполнения задания нами предоставляется исходный код в составе:
- constellationTest.json – конфигурационный файл с данными для инициализации орбитальной конфигурации космической системы
- gatewaysTest.json – файл с координатами расположения шлюзовых станций
- Constellation.m (или constellation.py) – файл, реализующий класс Constellation (орбитальная группировка), 
- example.m (или ConstellationExample.py) – пример работы с объектом типа Constellation

## Для справки ##
При вычислении зон обзора поверхности Земли космическим аппаратом (КА) используются следующие параметры и обозначения (рис. 1):
α – угол обзора (половина угла раствора конуса с вершиной в точке S и центральной осью, проходящей через центр Земли),
ε – угол места.

![alt text](./coverage.png "Рис. 1 - базовые геометрические параметры покрытия")

Во всех заданиях ниже поверхность Земли считается идеальной сферой с неподвижным центром. Ось вращения Земли совпадает с осью z принятой инерциальной системы координат с началом в центре Земли. Нулевой меридиан на начальную эпоху принадлежит плоскости xz неподвижной системы координат.

Орбитальная конфигурация системы состоит из двух групп спутников (каждая группа имеет свою высоту орбиты). Внутри каждой группы КА принадлежат к нескольким орбитальным плоскостям. Плоскости равномерно распределены по долготе, а КА равномерно распределены по окружностям орбит в каждой из плоскостей.

## Задание: ##

	1. Code review – предложите правки по логике и оформлению предоставленного вам кода (StyleGuide прилагается. Он в большей степени относится к MATLAB, но есть и независимые от языка соображения. При выполнении задания на python это руководство можно трактовать более вольно :)

	2. Реализуйте вычислительную процедуру, проверяющую, обеспечивает ли выбранная орбитальная конфигурация глобальное покрытие поверхности Земли в зависимости от значения угла α (ограничения на угол места отсутствуют).
    Примените реализованную процедуру для орбитальной конфигурации, соответствующей constellationTest.json в начальный момент времени. Найдите минимальное значение α, при котором глобальное покрытие обеспечивается (точность – 3 градуса).

	3. В файле gatewaysTest заданы координаты шлюзовых станций, расположенных на Земле. Реализуйте вычислительную процедуру, позволяющую для набора произвольных моментов времени определить, какие КА находятся в зоне видимости шлюзовых станций (группировку использовать из файла constellationTest.json). Условие нахождения КА в зоне видимости шлюзовой станции – угол места ε ≥ 25° (ограничения на α не накладываются). Для каждой пары 'шлюзовая станция' + КА в каждый момент времени, когда возможна связь, определите: ориентацию антенны (azimuth-elevation), расстояние между шлюзовой станицей и КА, допплеровский сдвиг частоты радиосигнала (несущая частота 433МГц). 

	4. Предположим, что КА системы (используйте группировку заданную в constellationTest.json) оснащены межспутниковой линией связи (МЛС), которая устроена так, что информация может передаваться от КА к соседним КА в его орбитальной плоскости. Воспользовавшись набором шлюзовых станций (файл gatewaysTest), считая, что связь КА со шлюзовой станцией возможна, если угол места КА ε ≥ 25°, определите наличие на поверхности Земли зон, не связанных спутниковой сетью (то есть существование точек на поверхности Земли, которые не могут быть обслужены КА, имеющими возможность пересылать информацию на шлюзовые станции). Значение угла обзора равно α=40°. Расчёт проведите на произвольный фиксированный момент времени.
	
	5. В файле testDataFile.mat содержатся данные о координатах некоторых точек на Земле (переменная coordsEcef, массив координат точек в осях связанных с вращающейся Землёй) и о значении некоторого параметра в каждой точке (массив values, параметр имеет смысл абонентского трафика). Требуется для любого состояния космических аппаратов (КА) системы (орбитальная конфигурация определяется конфигурационным файлом constellationTest.json, состояние - положения всех КА на произвольный момент времени) сформировать зоны обслуживания всех КА. Под зоной обслуживания будем понимать привязку каждой точки из набора coordsEcef к конкретному КА. Предлагается выбирать КА, в зону обслуживания которого входит точка X, на основании минимума расстояния от X до подспутниковой точки КА. Ответом к задаче является вычислительная процедура, распределяющая точки по зонам обслуживания (и создающая структуру данных, хранящую это распределение), а также массив с сумарным значением трафика в зонах обслуживания. 

	6. В файле testDataFile.mat содержатся данные о координатах абонентов на Земле (переменная coordsEcef, массив координат точек в осях связанных с вращающейся Землёй) и о значении некоторого параметра в каждой точке (массив values, параметр имеет смысл запрашиваемого абонентами трафика). Требуется для состояния космических аппаратов (КА), вычисленного на произвольный фиксированный момент времени, построить график зависимости удовлетворённого спроса от угла α. Абонентский спрос считается удовлетворённым, если точка на Земле, запрашивающая трафик, находится в зоне радиовидимости хотя бы одного КА системы. Зоны радиовидимости определяются значением угла α. Структура группировки определена в constellationTest.json Примечание: задача проверяется с использованием распределения абонентов и запросного трафика, отличающихся от предложенного в файле testDataFile.mat
	
	7. Предложите вычислительную процедуру построения матрицы связности для межспутниковой передачи данных (элемент ij имеет значение true, если связь между КА с номерами i и j есть, false - если нет). Постройте матрицу для одного эшелона (высота 720 км) группировки, которая создаётся в примере. Рассмотрите два случая: а) межспутниковая связь у каждого КА может быть с двумя соседними КА в той же орбитальной плоскости; б) то же, что в предыдущем случае, плюс у каждого КА может быть одна связь с соседней плоскостью. В последнем случае реализуйте процедуру поиска кратчайшего пути (минимум узлов) между двумя произвольно выбранными КА.
	
	8. Реализуйте вычислительную процедуру, позволяющую равномерно расположить N точек на поверхности сферической Земли. Воспользовавшись этой процедурой, разделите поверхность Земли на 64000 примерно одинаковых областей (сот). Воспользовавшись той же процедурой, населите поверхность Земли абонентами (2500000 равномерно распределённых по поверхности абонентов). Ассоциируйте абонентов с сотами (создайте структуру данных, в которой содержится информация о принадлежности каждого абонента к одной из сот). Для некоторого состояния космической системы (координаты КА, соответствующие группировке 'Starlink' из constellationTest.json и некоторому фиксированному моменту времени, вычисление координат реализовано в примере) распределите соты между КА системы (создайте структуру данных, в которой содержится информация о том, каким КА обслуживается каждая сота).

 	9. Реализуйте вычислительную процедуру, позволяющую равномерно расположить N точек на поверхности сферической Земли. Воспользовавшись этой процедурой, разделите поверхность Земли на 64000 примерно одинаковых области (соты). Для некоторого состояния космической системы (координаты КА, соответствующие группировке 'Starlink' из constellationTest.json и некоторому фиксированному моменту времени, вычисление координат реализовано в примере) определите для каждой соты набор КА, которые удовлетворяют следующему условию: угол между направлением от центра соты на любой видимый из неё геостационарный спутник составляет с направлением от центра соты на КА не более 2 градусов. Для простоты будем считать, что геостационарных спутников всего 4 и они равномерно распределены по геостационарной орбите. Создайте структуру данных, в которой содержится список таких КА для каждой соты. 
