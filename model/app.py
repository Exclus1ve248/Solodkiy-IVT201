from requests import get
height = input('Введите height = ')
weight = input('Введите weight = ')
gender = input('Введите gender(м-1;ж-0) = ')
print(get(f'http://127.0.0.1:5000/api', json={'height':height, 'weight':weight, 'gender':gender}).json())