# 1.- Importar las bibliotecas de Pandas

import pandas as pd

# 2.- Cargar los datos desde la URL:
#
# url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv'
# #porte está separado por tabulaciones

url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv'
chipo = pd.read_csv(url, sep='\t')

# 3.- Imprime sólo los 10 primeros resultados

print('Imprime sólo los 10 primeros resultados')
print(chipo.head(10))

print(chipo[['item_name', 'quantity']])

# 4.- Imprime el número de resultados disponibles (filas)

print('\nImprime el número de resultados disponibles (filas)', chipo.shape[0])

# 5.- Imprime el número de resultados disponibles (columnas)

print('\nImprime el número de resultados disponibles (columnas)', chipo.shape[1])

filas, columnas = chipo.shape

print('\n Filas:', filas, 'columnas:', columnas)

# 6.- Imprime el nombre de cada columna

print('\nImprime el nombre de cada columna')
print(chipo.columns)

# 7.- ¿Cómo se indexa el dataset?

print('\n¿Cómo se indexa el dataset?')
print(chipo.index)

# 8.- ¿Cuál es el item más pedido por pedido?

print(chipo.groupby(['order_id', 'item_name']).sum())

# 9.- ¿Cuántos items se han pedido?

items = chipo.groupby('item_name').sum()
print('¿Cuántos items diferentes se han pedido?', items.shape[0])
print('¿Cuántos items se han pedido?')
print(items['quantity'])

# 10.- ¿Cuál es el item más pedido por la columna choice_description?

print('\n¿Cuál es el item más pedido por la columna choice_description?')
items = chipo.groupby('choice_description').sum().sort_values('quantity', ascending=False)
print(items['quantity'].head(1))

# 11.- ¿Cuántos items se han ordenado en total?

print('¿Cuántos items se han ordenado en total?', chipo.sum()['quantity'])

# 12.- Convierte el precio en un float

dollarizer = lambda x: float(x[1:-1])
chipo.item_price = chipo.item_price.apply(dollarizer)


# 13.- ¿Cuantos ingresos se han tenido en total?

total = chipo.quantity * chipo['item_price']

print('¿Cuantos ingresos se han tenido en total?', total.sum())

# 14.- ¿Cuántos pedidos se han hecho en total?

items = chipo.groupby('order_id').count()
print('¿Cuántos pedidos se han hecho en total?', items.shape[0])

# 15.- ¿Cuánta es la cantidad promedio por pedido?

print('¿Cuánta es la cantidad promedio por pedido?', chipo.groupby('order_id').sum().mean().quantity)

# 16.- ¿Cuántos items diferentes se han pedido?

# Ver linea 43
