#!/usr/bin/env python
# coding: utf-8

# <center>
#     <h1>Máster en Business Analytics y Big Data (2019 - 2020)</h1>
#     <h3><i>Trabajo de Fin de Máster</i></h3>
#     <h2>"Predicción de ventas y detección de anomalías en tiempos de COVID-19 usando datos del mercado de valores y de la prensa internacional"</h2>
#     <h3>Autor: Sergio Carrasco Sánchez</h3>
#     <h3>Tutor: Juan Manuel Moreno Lamparero</h3>
# </center>

# <h2>Carga y limpieza de datos</h2>
# <h3>Carga de datos de ventas</h3>

# In[1]:


get_ipython().system('pip install pandas')

import pandas as pd


# Cargamos datos de ventas desde archivo CSV.

# In[2]:


sales_data_df = pd.read_csv('data/pedidos.csv')


# In[3]:


sales_data_df


# Eliminamos las columnas menos interesantes.

# In[4]:


sales_data_df = sales_data_df.drop(['id', 'usuario', 'forma_pago', 'subtotal', 'descuento', 'gastos_envio'], axis=1)


# In[5]:


sales_data_df = sales_data_df.rename({'fecha': 'date', 'total': 'sales'}, axis=1)


# In[6]:


sales_data_df


# Agrupamos las ventas por día.

# In[7]:


def daily_sales(data):
    data = data.copy()
    data.date = data.date.apply(lambda x: str(x)[:-9])
    data = data.groupby('date')['sales'].sum().reset_index()
    data.date = pd.to_datetime(data.date)
    
    return data


# In[8]:


daily_sales_df = daily_sales(sales_data_df)


# In[9]:


daily_sales_df


# In[10]:


daily_sales_df = daily_sales_df.set_index(daily_sales_df.date)


# In[11]:


daily_sales_df = daily_sales_df.drop('date', axis=1)


# In[12]:


daily_sales_df


# <h3>Carga de datos del índice Nasdaq Composite</h3>

# Importamos los datos del índice <strong>NASDAQ Composite</strong> del mismo periodo de tiempo de los datos de ventas.

# In[13]:


get_ipython().system('pip install yfinance')

import yfinance as yf


# In[14]:


min_date = min(daily_sales_df.index)
max_date = max(daily_sales_df.index)


# In[15]:


min_date


# In[16]:


max_date


# In[17]:


ticker = '^IXIC'
ticker_name = 'NASDAQ Composite'


# In[18]:


stock_data_df = yf.download(ticker, start=min_date, end=max_date)


# In[19]:


stock_data_df.index.name = 'date'


# In[20]:


stock_data_df


# Unimos en un mismo dataframe los datos de ventas y los del índice.

# In[21]:


daily_sales_df = pd.merge(left=daily_sales_df,
                          right=stock_data_df[{'Open',
                                               'High',
                                               'Low',
                                               'Close',
                                               'Adj Close',
                                               'Volume'}],
                          left_index=True,
                          right_index=True,
                          how='inner')


# In[22]:


daily_sales_df = daily_sales_df.rename({'Open':'stock_open',
                                        'High':'stock_high',
                                        'Low':'stock_low',
                                        'Close':'stock_close',
                                        'Adj Close':'stock_adjclose',
                                        'Volume':'stock_volume'},
                                       axis=1)


# In[23]:


daily_sales_df = daily_sales_df[['sales',
                                 'stock_open',
                                 'stock_high',
                                 'stock_low',
                                 'stock_close',
                                 'stock_adjclose',
                                 'stock_volume']]


# In[24]:


daily_sales_df


# <h3>Carga de datos del diario digital "The Economic Times"</h3>

# In[25]:


get_ipython().system('pip install requests')
get_ipython().system('pip install beautifulsoup4')

import requests
from bs4 import BeautifulSoup
import time
import datetime
from dateutil import rrule
from calendar import monthrange
import csv


# In[26]:


def read_url(year, month, starttime):
    url = f'https://economictimes.indiatimes.com/archivelist/year-{year},month-{month},starttime-{starttime}.cms'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    return soup


# In[27]:


def get_starttime(year, month, day):
    date1 = '1899-12-30'
    timestamp1 = time.mktime(datetime.datetime.strptime(date1, '%Y-%m-%d').timetuple())
    
    date2 = str(year) + '-' + str(month) + '-' + str(day)
    timestamp2 = time.mktime(datetime.datetime.strptime(date2, '%Y-%m-%d').timetuple())
    
    starttime = ((timestamp2 - timestamp1) / 86400)
    
    return str(starttime).replace(".0", "")


# In[28]:


headlines_from = '2020-01-01'
headlines_to = '2020-10-31'


# In[30]:


headlines_datetime_from = datetime.datetime.strptime(headlines_from, '%Y-%m-%d')
headlines_datetime_to = datetime.datetime.strptime(headlines_to, '%Y-%m-%d')


# In[ ]:


for dt in rrule.rrule(rrule.MONTHLY, dtstart=headlines_datetime_from, until=headlines_datetime_to):
    year = int(dt.strftime('%Y'))
    month = int(dt.strftime('%m'))
    
    for day in range(1, (monthrange(year, month)[1] + 1)):
        starttime = get_starttime(year, month, day)
        date_str_eng = str(year) + '-' + '{:02d}'.format(month) + '-' + '{:02d}'.format(day)
        
        #print(f'Date: {year}-{month}-{day}')
        
        headlines = []

        soup = read_url(year, month, starttime)

        for td in soup.findAll('td', {'class':'contentbox5'}):
            for headline in td.findAll('a'):
                if 'archive' not in headline.get('href'):
                    if len(headline.contents) > 0:
                        if headline.contents[0] not in headlines:
                            headlines.append(headline.contents[0])

        time.sleep(1)

        file = open(f'data/economic_news_headlines_{date_str_eng}.csv', 'w')
        with file:
            write = csv.writer(file, escapechar='\\', quoting=csv.QUOTE_NONE)
            for item in headlines:
                write.writerow([item,])


# Detectamos las palabras negativas encontradas en los titulares de noticias económicas.

# In[31]:


get_ipython().system('pip install stop-words')

from stop_words import get_stop_words
import collections


# In[32]:


stop_words = get_stop_words('en')


# In[33]:


banned_chars = ['\\', '`', '"', '*', '_', '{', '}', '[', ']', '(', ')', '>', '#',
                '+', ':', '-', '.', ',', '¿', '?', '¡', '!', '$', '\'', '«', '»', '|']


# In[34]:


negative_economic_words = ['coronavirus', 'sars-cov-2', 'covid-19', 'covid19', 'virus', 'pandemic',
                           'lockdown', 'outbreak', 'curfew', 'quarantine', 'crisis', 'fears', 'violence',
                           'death', 'cases', 'fall', 'hit', 'impact']


# In[35]:


number_common_words = 20


# In[36]:


negative_economic_words_df = pd.DataFrame(columns=['date', 'negative_economic_words'])


# In[37]:


date_from = '2020-01-01'
date_to = '2020-10-31'


# In[38]:


for date in pd.date_range(date_from, date_to, freq='d'):
    date_str_eng = date.strftime('%Y-%m-%d')
    
    print(f'Date: {date_str_eng}')
    print()
    
    file = open(f'data/economic_news_headlines_{date_str_eng}.csv', 'rt')

    headlines = []
    
    with file:
        csv_reader = csv.reader(file, escapechar='\\')

        for line in csv_reader:
            headlines.append(line)
    
    word_count = {}
    
    for headline in headlines:
        for word in headline[0].lower().split():
            for ch in banned_chars:
                if ch in word:
                    word = word.replace(ch, '')

            if (word != '') & (word not in stop_words):
                if word not in word_count:
                    word_count[word] = 1
                else:
                    word_count[word] += 1
    
    negative_words_count = 0
    word_counter = collections.Counter(word_count)
    
    print(f"Top {number_common_words} most common words:")
    print()
    
    for word, count in word_counter.most_common(number_common_words):
        print(f'{word}: {count}')

        if word in negative_economic_words:
            negative_words_count += count
    
    print()
    print(f"Negative words: {negative_words_count}")
    print()
    
    negative_economic_words_df = negative_economic_words_df.append({'date':date,
                                                                    'negative_economic_words':negative_words_count},
                                                                   ignore_index=True)


# In[39]:


negative_economic_words_df


# In[40]:


negative_economic_words_df.to_csv('data/negative_economic_words.csv', index=False)


# Creamos una nube de palabras con todas las palabras negativas extraídas de los titulares de prensa.

# In[41]:


headlines = []


# In[42]:


for date in pd.date_range(date_from, date_to, freq='d'):
    date_str_eng = date.strftime('%Y-%m-%d')
    
    file = open(f'data/economic_news_headlines_{date_str_eng}.csv', 'rt')

    with file:
        csv_reader = csv.reader(file, escapechar='\\')

        for line in csv_reader:
            headlines.append(line)


# In[43]:


word_count = {}
negative_word_count = {}

for headline in headlines:
    for word in headline[0].lower().split():
        for ch in banned_chars:
            if ch in word:
                word = word.replace(ch, '')
        
        if (word != '') & (word not in stop_words):
            if word not in word_count:
                word_count[word] = 1
            else:
                word_count[word] += 1
            
            if word in negative_economic_words:
                if word not in negative_word_count:
                    negative_word_count[word] = 1
                else:
                    negative_word_count[word] += 1


# In[44]:


number_common_words = 25

word_counter = collections.Counter(word_count)
negative_word_counter = collections.Counter(negative_word_count)

most_common_words = {}
most_common_negative_words = {}

for word, count in word_counter.most_common(number_common_words):
    most_common_words[word] = count

for word, count in negative_word_counter.most_common(number_common_words):
    most_common_negative_words[word] = count
    print(f'{word}: {count}')


# In[45]:


get_ipython().system('pip install wordcloud')
get_ipython().system('pip install matplotlib')

from wordcloud import WordCloud
import matplotlib.pyplot as plt


# In[46]:


wc = WordCloud(background_color='white',
               max_font_size=256,
               random_state=42,
               width=800,
               height=400
              ).generate_from_frequencies(most_common_negative_words)
plt.figure(figsize=(12, 8))
plt.imshow(wc, interpolation="bilinear")
plt.axis('off')
plt.show()


# Creamos el dataframe "negative_economic_words_df" que contiene el número de palabras negativas de la prensa económica desde enero a octubre de 2020, y se unen estos datos al dataframe "daily_sales_df" que ya contenía los valores de las ventas y del índice Nasdaq Composite del mismo periodo de tiempo.

# In[47]:


negative_economic_words_csv_df = pd.read_csv('data/negative_economic_words.csv')


# In[48]:


negative_economic_words_csv_df = negative_economic_words_csv_df.set_index(negative_economic_words_csv_df.date)


# In[49]:


negative_economic_words_csv_df = negative_economic_words_csv_df.drop('date', axis=1)


# In[50]:


negative_economic_words_df = pd.DataFrame(columns=['date'])


# In[51]:


for date in pd.date_range(min_date, max_date, freq='d'):
    negative_economic_words_df = negative_economic_words_df.append({'date':date}, ignore_index=True)


# In[52]:


negative_economic_words_df = negative_economic_words_df.set_index(negative_economic_words_df.date)


# In[53]:


negative_economic_words_df = negative_economic_words_df.drop('date', axis=1)


# In[54]:


negative_economic_words_df = pd.merge(left=negative_economic_words_df,
                                      right=negative_economic_words_csv_df[{'negative_economic_words'}],
                                      left_index=True,
                                      right_index=True,
                                      how='outer')


# In[55]:


negative_economic_words_df = negative_economic_words_df.fillna(0)


# In[56]:


daily_sales_df = pd.merge(left=daily_sales_df,
                          right=negative_economic_words_df[{'negative_economic_words'}],
                          left_index=True,
                          right_index=True,
                          how='inner')


# In[57]:


daily_sales_df


# Comprobamos el contraste entre los valores de las columnas "sales" y "negative_economic_words", principalmente a partir del día 13 de marzo.

# In[58]:


daily_sales_df[(daily_sales_df.index >= '2020-03-01') & (daily_sales_df.index <= '2020-03-31')]


# <h3>Carga de datos del diario digital ABC.es</h3>

# In[59]:


def read_url(date, page):
    url = f'https://www.abc.es/hemeroteca/dia-{date}/pagina-{page}'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    html = soup.text
    contents = not("Sugerencias" in html)
    
    return contents, soup


# In[60]:


headlines_from = '2020-05-21'
headlines_to = '2020-10-31'


# In[62]:


headlines_timestamp_from = datetime.datetime.strptime(headlines_from, '%Y-%m-%d')
headlines_timestamp_to = datetime.datetime.strptime(headlines_to, '%Y-%m-%d')


# In[ ]:


for date in pd.date_range(headlines_timestamp_from, headlines_timestamp_to, freq='d'):
    date_str_eng = date.strftime('%Y-%m-%d')
    date_str_spa = date.strftime('%d-%m-%Y')
    
    #print(f'Date: {date_str_eng}')
    #print()
    
    headlines = []
    page = 1
    
    contents, soup = read_url(date_str_spa, page)
    
    while contents:
        #print(f'Page: {page}')
        #print()
        for headline in soup.findAll('a', {'class':'titulo'}):
            #print(headline.contents[0])
            headlines.append(headline.contents[0])            
        time.sleep(1)
        page += 1
        contents, soup = read_url(date_str_spa, page)
        #print()
    
    file = open(f'data/news_headlines_{date_str_eng}.csv', 'w')
    
    with file:
        write = csv.writer(file, escapechar='\\', quoting=csv.QUOTE_NONE)
        for item in headlines:
            write.writerow([item,])


# Se construyen las listas "stop_words" y "banned_chars" con las palabras (en español) y símbolos a evitar, y "negative_words" con las palabras negativas (en español) más comunes a contabilizar.

# In[63]:


stop_words = get_stop_words('es')


# In[64]:


banned_chars = ['\\', '`', '"', '*', '_', '{', '}', '[', ']', '(', ')', '>', '#',
                '+', ':', '-', '.', ',', '¿', '?', '¡', '!', '$', '\'', '«', '»', '|']


# In[65]:


negative_words = ['coronavirus', 'sars-cov-2', 'covid-19', 'covid19', 'crisis', 'recesión', 'quiebra', 'caída',
                  'erte', 'ertes', 'cierre', 'cierra', 'ruina', 'alarma', 'medidas', 'casos', 'cuarentena', 'confinamiento',
                  'colapso', 'contagios', 'pandemia', 'epidemia', 'muertos', 'muertes', 'muere', 'fallecidos']


# In[66]:


number_common_words = 25


# In[67]:


negative_words_df = pd.DataFrame(columns=['date', 'negative_words'])


# In[68]:


date_from = '2020-01-01'
date_to = '2020-10-31'


# In[69]:


for date in pd.date_range(date_from, date_to, freq='d'):
    date_str_eng = date.strftime('%Y-%m-%d')
    
    print(f'Date: {date_str_eng}')
    print()
    
    file = open(f'data/news_headlines_{date_str_eng}.csv', 'rt')

    headlines = []
    
    with file:
        csv_reader = csv.reader(file, escapechar='\\')

        for line in csv_reader:
            headlines.append(line)
    
    word_count = {}
    
    for headline in headlines:
        for word in headline[0].lower().split():
            for ch in banned_chars:
                if ch in word:
                    word = word.replace(ch, '')

            if (word != '') & (word not in stop_words):
                if word not in word_count:
                    word_count[word] = 1
                else:
                    word_count[word] += 1
    
    negative_words_count = 0
    word_counter = collections.Counter(word_count)
    
    print(f"Top {number_common_words} most common words:")
    print()
    
    for word, count in word_counter.most_common(number_common_words):
        print(f'{word}: {count}')

        if word in negative_words:
            negative_words_count += count
    
    print()
    print(f"Negative words: {negative_words_count}")
    print()
    
    negative_words_df = negative_words_df.append({'date':date, 'negative_words':negative_words_count}, ignore_index=True)


# In[70]:


negative_words_df


# In[71]:


negative_words_df.to_csv('data/negative_words.csv', index=False)


# Realizamos una nube de palabras con todas las palabras negativas extraídas de los titulares de prensa.

# In[72]:


headlines = []


# In[73]:


for date in pd.date_range(date_from, date_to, freq='d'):
    date_str_eng = date.strftime('%Y-%m-%d')
    
    file = open(f'data/news_headlines_{date_str_eng}.csv', 'rt')

    with file:
        csv_reader = csv.reader(file, escapechar='\\')

        for line in csv_reader:
            headlines.append(line)


# In[74]:


word_count = {}
negative_word_count = {}

for headline in headlines:
    for word in headline[0].lower().split():
        for ch in banned_chars:
            if ch in word:
                word = word.replace(ch, '')

        if (word != '') & (word not in stop_words):
            if word not in word_count:
                word_count[word] = 1
            else:
                word_count[word] += 1
            
            if word in negative_words:
                if word not in negative_word_count:
                    negative_word_count[word] = 1
                else:
                    negative_word_count[word] += 1


# In[75]:


number_common_words = 25

word_counter = collections.Counter(word_count)
negative_word_counter = collections.Counter(negative_word_count)

most_common_words = {}
most_common_negative_words = {}

for word, count in word_counter.most_common(number_common_words):
    most_common_words[word] = count

for word, count in negative_word_counter.most_common(number_common_words):
    most_common_negative_words[word] = count
    print(f'{word}: {count}')


# In[76]:


wc = WordCloud(background_color='white',
               max_font_size=256,
               random_state=42,
               width=800,
               height=400
              ).generate_from_frequencies(most_common_negative_words)
plt.figure(figsize=(12, 8))
plt.imshow(wc, interpolation="bilinear")
plt.axis('off')
plt.show()


# Por último, se crea el dataframe "negative_words_df" que contiene el número de palabras negativas de la prensa generalista desde enero a octubre de 2020, y se unen estos datos al dataframe "daily_sales_df" que ya contenía los valores de las ventas, del índice Nasdaq Composite y el número de palabras negativas de la prensa económica del mismo periodo de tiempo.

# In[77]:


negative_words_csv_df = pd.read_csv('data/negative_words.csv')


# In[78]:


negative_words_csv_df = negative_words_csv_df.set_index(negative_words_csv_df.date)


# In[79]:


negative_words_csv_df = negative_words_csv_df.drop('date', axis=1)


# In[80]:


negative_words_df = pd.DataFrame(columns=['date'])


# In[81]:


for date in pd.date_range(min_date, max_date, freq='d'):
    negative_words_df = negative_words_df.append({'date':date}, ignore_index=True)


# In[82]:


negative_words_df = negative_words_df.set_index(negative_words_df.date)


# In[83]:


negative_words_df = negative_words_df.drop('date', axis=1)


# In[84]:


negative_words_df = pd.merge(left=negative_words_df,
                             right=negative_words_csv_df[{'negative_words'}],
                             left_index=True,
                             right_index=True, how='outer')


# In[85]:


negative_words_df = negative_words_df.fillna(0)


# In[86]:


daily_sales_df = pd.merge(left=daily_sales_df,
                          right=negative_words_df[{'negative_words'}],
                          left_index=True,
                          right_index=True,
                          how='inner')


# Comprobamos el contraste entre los valores de las columnas “sales” y “negative_words”, principalmente a partir del día 12 de marzo.

# In[87]:


daily_sales_df[(daily_sales_df.index >= '2020-03-01') & (daily_sales_df.index <= '2020-03-31')]


# <h2>Análisis exploratorio de datos</h2>

# In[88]:


daily_sales_df.info()


# In[89]:


daily_sales_df.describe()


# Mediante la observación de los histogramas ya se aprecia una fuerte correlación entre las variables "stock_open", "stock_high", "stock_low", "stock_close" y "stock_adjclose", así como entre las variables "negative_economic_words" y "negative_words".

# In[90]:


daily_sales_df.hist(figsize=(15, 15))


# Se procede a realizar algunas comprobaciones adicionales sobre los datos.

# In[91]:


get_ipython().system('pip install pandas-profiling')

from pandas_profiling import ProfileReport


# In[92]:


profile = ProfileReport(daily_sales_df, title="Pandas Profiling Report")


# In[93]:


profile


# A continuación, se profundiza en el estudio de la correlación entre variables mediante la creación de un mapa de calor.

# In[94]:


get_ipython().system('pip install seaborn')

import seaborn as sns


# Comprobamos la posible correlación entre variables mediante un mapa de calor, para lo cual utilizamos el método corr de Pandas.

# In[95]:


corr = daily_sales_df.corr()


# In[96]:


plt.subplots(figsize=(10, 10))
sns.heatmap(corr, annot=True, cmap='YlGnBu')
plt.show()


# Como podemos observar, el mapa de calor de correlación nos proporciona una descripción visual de la relación entre las variables. Ahora bien, no queremos un conjunto de variables independientes que tenga una relación más o menos similar con las variables dependientes. Si nos fijamos en la variable dependiente “sales” nos cercioramos que no existe fuerte correlación con ninguna variable independiente.

# No obstante, se tratará de eliminar la fuerte dependencia entre las variables independientes "stock_open", "stock_high", "stock_low", "stock_close" y "stock_adjclose" añadiendo una nueva variable llamada "stock_mean" como media de las variables "stock_high" y "stock_low", y se comprueba de nuevo la correlación resultante.

# In[97]:


daily_sales_df['stock_mean'] = (daily_sales_df.stock_low + daily_sales_df.stock_high) / 2


# In[98]:


daily_sales_df = daily_sales_df[['sales', 'stock_volume', 'stock_mean', 'negative_words', 'negative_economic_words']]


# In[99]:


corr = daily_sales_df.corr()


# In[100]:


plt.subplots(figsize=(10, 10))
sns.heatmap(corr, annot=True, cmap='YlGnBu')
plt.show()


# Aunque se sigue observando una fuerte correlación entre las variables "negative_words" y "negative_economic_words" se mantendrán para la aplicación del modelo predictivo.

# Observamos en una gráfica el comportamiento de las ventas diarias.

# In[101]:


figsize=(18, 8)
plt.figure(figsize=figsize)
plt.title('Daily Sales')
plt.plot(daily_sales_df.index, daily_sales_df.sales)
plt.show()


# Observamos el comportamiento de los valores del índice NASDAQ.

# In[102]:


figsize=(18, 8)
plt.figure(figsize=figsize)
plt.title(f'{ticker_name} Low & High Mean')
plt.plot(daily_sales_df.index, daily_sales_df.stock_mean)
plt.show()


# Comprobamos que tiene un comportamiento ciertamente similar al de las ventas.

# Comprobamos que a mediados de marzo de 2020 se produce un fuerte descenso de las ventas, y un fuerte aumento un mes después, con un crecimiento sostenido en el futuro respecto a periodos anteriores.

# A continuación, se observan los mismos valores en un periodo de tiempo que contiene el momento inicial de la pandemia (entre enero y mayo de 2020), comprobando que tienen un comportamiento ciertamente similar.

# In[103]:


pandemic_period_date_from = '2020-01-01'
pandemic_period_date_to = '2020-05-31'


# In[104]:


figsize=(18, 8)
plt.figure(figsize=figsize)
plt.title('Pandemic Sales')
plt.plot(daily_sales_df.index[(daily_sales_df.index >= pandemic_period_date_from) & (daily_sales_df.index <= pandemic_period_date_to)], daily_sales_df['sales'][(daily_sales_df.index >= pandemic_period_date_from) & (daily_sales_df.index <= pandemic_period_date_to)])
plt.show()


# In[105]:


figsize=(18, 8)
plt.figure(figsize=figsize)
plt.title(f'Pandemic {ticker_name} Low & High Mean')
plt.plot(daily_sales_df.index[(daily_sales_df.index >= pandemic_period_date_from) & (daily_sales_df.index <= pandemic_period_date_to)], daily_sales_df.stock_mean[(daily_sales_df.index >= pandemic_period_date_from) & (daily_sales_df.index <= pandemic_period_date_to)])
plt.show()


# Por último, se observa la gráfica de los valores de las palabras negativas, comprobando que tiene un comportamiento inverso al de las ventas y del índice Nasdaq Composite.

# In[106]:


figsize=(18, 8)
plt.figure(figsize=figsize)
plt.title('Negative Words')
plt.plot(daily_sales_df.index[(daily_sales_df.index >= pandemic_period_date_from) & (daily_sales_df.index <= pandemic_period_date_to)], daily_sales_df['negative_words'][(daily_sales_df.index >= pandemic_period_date_from) & (daily_sales_df.index <= pandemic_period_date_to)])
plt.show()


# <h2>Modelo de predicción</h2>

# Se procede a la preparación de los conjuntos de datos de entrenamiento y de prueba con las variables independientes (exógenas) y dependientes (endógenas) para nuestro modelo predictivo de series temporales SARIMAX.

# In[107]:


daily_sales_df = daily_sales_df.reset_index()


# In[108]:


daily_sales_df


# Preparamos el dataset <strong>X</strong> con las variables independientes (exógenas).

# In[109]:


X = daily_sales_df[['date', 'stock_volume', 'stock_mean', 'negative_words', 'negative_economic_words']]


# In[110]:


X


# Preparamos el dataset <strong>y</strong> con la variable dependiente (endógena).

# In[111]:


y = daily_sales_df[['date', 'sales']]


# In[112]:


y


# Preparamos los dataset <strong>train_sales_vals</strong> y <strong>train_exog_vals</strong> con los conjuntos de datos de entrenamiento desde el periodo 01/01/2016 a 31/12/2019. Desestimamos las ventas anteriores a 01/01/2016 al presentar valores de ventas irregulares (fueron los años de arranque del negocio).

# In[113]:


train_date_from = '2016-01-01'
train_date_to = '2019-12-31'


# In[114]:


train_size = int(len(X[(X.date >= train_date_from) & (X.date <= train_date_to)]))


# In[115]:


train_size


# In[116]:


train_sales_vals = y.sales[(y.date >= train_date_from) & (y.date <= train_date_to)]
train_sales_vals = train_sales_vals.reset_index(drop=True)


# In[117]:


train_exog_vals = X[(X.date >= train_date_from) & (X.date <= train_date_to)][['stock_volume', 'stock_mean', 'negative_words', 'negative_economic_words']]
train_exog_vals = train_exog_vals.reset_index(drop=True)


# Preparamos los dataset <strong>test_sales_vals</strong> y <strong>test_exog_vals</strong> con los conjuntos de datos de test desde el periodo 01/01/2020 a 30/10/2020.

# In[118]:


test_date_from = '2020-01-01'
test_date_to = '2020-10-30'


# In[119]:


test_size = int(len(y[(y.date >= test_date_from) & (y.date <= test_date_to)]))


# In[120]:


test_size


# In[121]:


test_sales_vals = y.sales[(y.date >= test_date_from) & (y.date <= test_date_to)]
test_sales_vals = test_sales_vals.reset_index(drop=True)


# In[122]:


test_exog_vals = X[(X.date >= test_date_from) & (X.date <= test_date_to)][['stock_volume', 'stock_mean', 'negative_words', 'negative_economic_words']]
test_exog_vals = test_exog_vals.reset_index(drop=True)


# A continuación, se procede a trabajar en el problema de predicción. Es importante recordar que, al tratarse de un problema de predicción de series temporales, es necesario probar la estacionalidad de la serie temporales para poder aplicar el modelo SARIMAX.

# Se parte de la observación de la estacionalidad de los valores de las ventas en el periodo de un año.

# In[123]:


get_ipython().system('pip install pip install statsmodels')

import statsmodels.api as sm


# In[124]:


seas_d = sm.tsa.seasonal_decompose(y['sales'], model='add', period=365)


# In[125]:


fig = seas_d.plot()
fig.set_figwidth(18)
fig.set_figheight(12)
plt.show()


# Comprobamos la estacionalidad de los valores de ventas en el periodo mensual.

# In[126]:


seas_d = sm.tsa.seasonal_decompose(y[y.date >= '2020-01-01']['sales'], model='add', period=30)


# In[127]:


fig = seas_d.plot()
fig.set_figwidth(18)
fig.set_figheight(12)
plt.show()


# Comprobamos la estacionalidad de los valores de ventas en el periodo semanal.

# In[128]:


seas_d = sm.tsa.seasonal_decompose(y[y.date >= '2020-01-01']['sales'], model='add', period=7)


# In[129]:


fig = seas_d.plot()
fig.set_figwidth(18)
fig.set_figheight(12)
plt.show()


# Se verifica que, al menos gráficamente, se observa una clara estacionalidad en los datos de ventas en los periodos anual, mensual y semanal.

# Si hacemos que los datos sean estacionarios, entonces el modelo puede hacer predicciones basadas en el hecho de que la media y la varianza seguirán siendo las mismas en el futuro. Una serie estacionaria es más fácil de predecir. Para comprobar si los datos son estacionarios, usaremos la prueba <strong>Augmented Dickey-Fuller (ADF)</strong>. Es el método estadístico más popular para encontrar si la serie es estacionaria o no. También se denomina prueba de raíz unitaria.

# In[130]:


get_ipython().system('pip install stattools')

from statsmodels.tsa.stattools import adfuller


# In[131]:


def test_adf(series, title=''):
    dfout={}
    dftest=sm.tsa.adfuller(series.dropna(), autolag='AIC', regression='ct')
    
    for key, val in dftest[4].items():
        dfout[f'critical value ({key})'] = val
    
    if dftest[1] <= 0.05:
        print('Strong evidence against Null Hypothesis')
        print('Reject Null Hypothesis - Data is Stationary')
        print('Data is Stationary for', title)
    else:
        print('Strong evidence for Null Hypothesis')
        print('Accept Null Hypothesis - Data is not Stationary')
        print('Data is NOT Stationary for', title)


# Comprobamos que los datos de ventas del conjunto de entrenamiento (del 01/01/2016 al 31/12/2019) son estacionarios.

# In[132]:


test_adf(train_sales_vals, 'Train Sales')


# Comprobamos que los datos de ventas del conjunto de test (del 01/01/2020 en adelante) no son estacionarios.

# In[133]:


test_adf(test_sales_vals, 'Test Sales')


# Comprobamos que los datos de ventas del conjunto de test (del 01/01/2020 en adelante) pasan a ser estacionarios aplicando la <strong>transformación logarítmica</strong>.

# In[134]:


get_ipython().system('pip install numpy')

import numpy as np


# In[135]:


test_adf(np.log10(test_sales_vals), 'Log10 Test Sales')


# Según esto, aplicamos la transformación logarítmica a todos los dataset para estabilizar la varianza en los datos y hacerla estacionaria antes de alimentarla al modelo.

# In[136]:


train_sales_log = np.log10(train_sales_vals)


# In[137]:


test_sales_log = np.log10(test_sales_vals)


# In[138]:


train_exog_log = np.log10(train_exog_vals)


# In[139]:


test_exog_log = np.log10(test_exog_vals)


# In[140]:


from numpy import inf


# In[141]:


train_exog_log[train_exog_log.negative_words == -inf] = 0


# In[142]:


test_exog_log[test_exog_log.negative_words == -inf] = 0


# Aplicamos ARIMA y SARIMAX a nuestros datos y veamos cuál funciona mejor. Tanto para ARIMA como para SARIMA o SARIMAX, necesitamos conocer los términos AR y MA para corregir cualquier autocorrelación en la serie diferenciada.

# Observamos los gráficos de función de autocorrelación (ACF) y de autocorrelación parcial (PACF) de la serie diferenciada.

# In[143]:


fig, ax = plt.subplots(2, 1, figsize=(18, 12))
fig=sm.tsa.graphics.plot_acf(train_sales_log, lags=50, ax=ax[0])
fig=sm.tsa.graphics.plot_pacf(train_sales_log, lags=50, ax=ax[1])
plt.show()


# In[144]:


fig, ax = plt.subplots(2, 1, figsize=(18, 12))
fig=sm.tsa.graphics.plot_acf(test_sales_log, lags=50, ax=ax[0])
fig=sm.tsa.graphics.plot_pacf(test_sales_log, lags=50, ax=ax[1])
plt.show()


# Vemos que el gráfico PACF tiene un pico significativo en el lag 1 y el lag 2, lo que significa que todas las autocorrelaciones de orden superior se explican de manera efectiva por las autocorrelaciones de lag 1 y lag 2.

# Usamos pyramid auto Arima para realizar una búsqueda paso a paso del término AR y MA que dé el valor más bajo de AIC.

# In[145]:


get_ipython().system('pip install pmdarima')

from pmdarima.arima import auto_arima


# In[146]:


stepwise_model = auto_arima(
    train_sales_log,
    exogenous=train_exog_log,
    start_p=0, start_q=0,
    start_P=0, start_Q=0,
    max_p=7, max_q=7,
    d=1, D=1,
    m=7,
    seasonal=True,
    trace=True,
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True)


# In[147]:


stepwise_model.summary()


# El modelo sugerido por auto_arima es SARIMAX, y los valores de p, d, q y de P, D, Q son 4, 1, 0 y 2, 1, 0, respectivamente.

# En este punto, se procede a predecir el siguiente punto de datos y se recorren los datos de entrenamiento para predecir los siguientes datos y agregar el siguiente punto de datos después de la predicción para un pronóstico adicional. Esto es como una ventana móvil de datos de nivel diario.

# In[148]:


import warnings

warnings.filterwarnings('ignore')


# In[149]:


predictions = list()
predict_log = list()


# In[150]:


with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    for t in range(len(test_sales_log)):
        stepwise_model.fit(train_sales_log)
        output = stepwise_model.predict(n_periods=1)
        predict_log.append(output[0])
        yhat = 10**output[0]
        predictions.append(yhat)
        obs = test_sales_log.iloc[t]
        train_sales_log = train_sales_log.append(pd.Series(obs), ignore_index=True)
        print('t=%f, predicted=%f, expected=%f' % (t, output[0], obs))
        obs = test_exog_log.iloc[t]
        train_exog_log = train_exog_log.append(pd.Series(obs), ignore_index=True)


# <h2>Resultados</h2>
# <h3>Visualización de gráfica de resultados</h3>

# Se procede a utilizar el error cuadrático medio (RMSE) para evaluar el modelo.

# In[151]:


get_ipython().system('pip install python-math')
get_ipython().system('pip install scikit-metrics')

import math
from sklearn.metrics import mean_squared_error


# In[152]:


error = math.sqrt(mean_squared_error(test_sales_log, predict_log[0:211:]))
print('Test RMSE: %.3f' % error)


# A continuación, para visualizar, creemos un marco de datos con los datos reales disponibles y los resultados de la predicción.

# In[153]:


predicted_df = pd.DataFrame()
predicted_df['date'] = daily_sales_df['date'][(daily_sales_df.date >= test_date_from) & (daily_sales_df.date <= test_date_to)]
predicted_df['sales'] = test_sales_vals.values
predicted_df['predicted'] = predictions


# In[154]:


predicted_df


# Dibujamos la gráfica de las predicciones en relación a las ventas reales.

# In[155]:


figsize=(18, 8)
plt.figure(figsize=figsize)
plt.title('Actual Sales vs Predicted Sales')
plt.plot(predicted_df.date, predicted_df.sales, label='Sales')
plt.plot(predicted_df.date, predicted_df.predicted, color='red', label='Predicted')
plt.legend(loc='upper right')
plt.show()


# Comprobamos que la predicción reproduce con bastante fidelidad el pico de caídas en las ventas de mediados de marzo, así como el repunte en ventas del mes siguiente.

# <h3>Detección y visualización de anomalías</h3>

# Una vez con los resultados de pronóstico y datos reales, procedemos a detectar anomalías. Para ello, se siguen los siguientes pasos:
# 
# 1. Cálculo del término de error (real-predicción).
# 2. Cálculo de la media móvil y la desviación estándar móvil (la ventana es una semana).
# 3. Clasificación de los datos con un error de 1.5, 1.75 y 2 desviaciones estándar como límites para anomalías bajas, medias y altas (el 5% de los puntos de datos serían anomalías identificadas según esta propiedad).

# In[156]:


def detect_classify_anomalies(df, window):
    df = df.copy()
    df.replace([np.inf, -np.inf], np.NaN, inplace=True)
    df.fillna(0, inplace=True)
    df['error'] = df['sales'] - df['predicted']
    df['percentage_change'] = ((df['error']) / df['sales']) * 100
    df['meanval'] = df['error'].rolling(window=window).mean()
    df['deviation'] = df['error'].rolling(window=window).std()
    df['-3s'] = df['meanval'] - (2 * df['deviation'])
    df['3s'] = df['meanval'] + (2 * df['deviation'])
    df['-2s'] = df['meanval'] - (1.75 * df['deviation'])
    df['2s'] = df['meanval'] + (1.75 * df['deviation'])
    df['-1s'] = df['meanval'] - (1.5 * df['deviation'])
    df['1s'] = df['meanval'] + (1.5 * df['deviation'])
    cut_list = df[['error', '-3s', '-2s', '-1s', 'meanval', '1s', '2s', '3s']]
    cut_values = cut_list.values
    cut_sort = np.sort(cut_values)
    df['impact'] = [(lambda x: np.where(cut_sort == df['error'].iloc[x])[1][0])(x) for x in range(len(df['error']))]
    severity = {0:3, 1:2, 2:1, 3:0, 4:0, 5:1, 6:2, 7:3}
    region = {0:'NEGATIVE', 1:'NEGATIVE', 2:'NEGATIVE', 3:'NEGATIVE', 4:'POSITIVE', 5:'POSITIVE', 6:'POSITIVE', 7:'POSITIVE'}
    df['color'] = df['impact'].map(severity)
    df['region'] = df['impact'].map(region)
    df['anomaly_points_level_1'] = np.where(df['color'] == 1, df['error'], np.nan)
    df['anomaly_points_level_2'] = np.where(df['color'] == 2, df['error'], np.nan)
    df['anomaly_points_level_3'] = np.where(df['color'] == 3, df['error'], np.nan)
    df = df.sort_values(by='date', ascending=False)
    df.date = pd.to_datetime(df['date'].astype(str), format='%Y-%m-%d')
    
    return df


# In[157]:


classify_df = detect_classify_anomalies(predicted_df, 7)


# In[158]:


classify_df.reset_index(inplace=True)


# In[159]:


classify_df = classify_df.drop("index", axis=1)


# In[160]:


classify_df.date = classify_df.date.dt.strftime('%Y-%m-%d')


# In[161]:


classify_df


# A continuación, se muestra una función para visualizar los resultados. Una vez más, la importancia de una visualización clara e integral ayuda a los usuarios comerciales a dar comentarios sobre las anomalías y hace que los resultados sean procesables.

# El primer gráfico tiene el término de error con el límite superior e inferior especificado, con las anomalías resaltadas sería fácil de interpretar/validar para un usuario. El segundo gráfico tiene valores reales y pronosticados con anomalías resaltadas.

# In[162]:


get_ipython().system('pip install plotly')

import plotly.graph_objects as go
from plotly.offline import iplot


# In[163]:


def plot_anomaly(df, metric_name):
    dates = df.date
    
    bool_array_level_1 = (abs(df['anomaly_points_level_1']) > 0)
    sales_level_1 = df["sales"][-len(bool_array_level_1):]
    anomaly_points_level_1 = bool_array_level_1 * sales_level_1
    anomaly_points_level_1[anomaly_points_level_1 == 0] = np.nan
    
    bool_array_level_2 = (abs(df['anomaly_points_level_2']) > 0)
    sales_level_2 = df["sales"][-len(bool_array_level_2):]
    anomaly_points_level_2 = bool_array_level_2 * sales_level_2
    anomaly_points_level_2[anomaly_points_level_2 == 0] = np.nan
    
    bool_array_level_3 = (abs(df['anomaly_points_level_3']) > 0)
    sales_level_3 = df["sales"][-len(bool_array_level_3):]
    anomaly_points_level_3 = bool_array_level_3 * sales_level_3
    anomaly_points_level_3[anomaly_points_level_3 == 0] = np.nan
    
    color_map = {0:'rgba(228, 222, 249, 0.65)', 1:'yellow', 2:'orange', 3:'red'}
    
    table = go.Table(
        domain = dict(x=[0, 1],
                      y=[0, 0.3]),
        columnwidth = [1, 2],
        header = dict(height=20,
                      values=[['<b>Date</b>'], ['<b>Sales</b>'],
                              ['<b>Predicted</b>'], ['<b>% Difference</b>'], ['<b>Severity (0-3)</b>']],
                      font=dict(color=['rgb(45, 45, 45)'] * 5, size=14),
                      fill=dict(color='#d562be')),
        cells = dict(values=[df.round(2)[k].tolist() for k in ['date', 'sales', 'predicted',
                                                               'percentage_change', 'color']],
                     line=dict(color='#506784'),
                     align=['center'] * 5,
                     font=dict(color=['rgb(40, 40, 40)'] * 5, size=12),
                     suffix=[None] + [''] + [''] + ['%'] + [''],
                     height=27,
                     fill=dict(color=[df['color'].map(color_map)])))
    
    error = go.Scatter(name='Error',
                       x=dates,
                       y=df['error'],
                       xaxis='x1',
                       yaxis='y1',
                       mode='lines',
                       marker=dict(size=12,
                                   line=dict(width=1),
                                   color='darkred'),
                       text='Error')
    
    mvingavrg = go.Scatter(name='Moving Average',
                           x=dates,
                           y=df['meanval'],
                           mode='lines',
                           xaxis='x1',
                           yaxis='y1',
                           marker=dict(size=12,
                                       line=dict(width=1),
                                       color='green'),
                           text='Moving average')
    
    anomalies_level_1 = go.Scatter(name='Anomaly Level 1',
                           x=dates,
                           xaxis='x1',
                           yaxis='y1',
                           y=df['anomaly_points_level_1'],
                           mode='markers',
                           marker=dict(color='yellow',
                                       size=11,
                                       line=dict(color='yellow',
                                                 width=1)))
    
    anomalies_level_2 = go.Scatter(name='Anomaly Level 2',
                           x=dates,
                           xaxis='x1',
                           yaxis='y1',
                           y=df['anomaly_points_level_2'],
                           mode='markers',
                           marker=dict(color='orange',
                                       size=11,
                                       line=dict(color='orange',
                                                 width=1)))
    
    anomalies_level_3 = go.Scatter(name='Anomaly Level 3',
                           x=dates,
                           xaxis='x1',
                           yaxis='y1',
                           y=df['anomaly_points_level_3'],
                           mode='markers',
                           marker=dict(color='red',
                                       size=11,
                                       line=dict(color='red',
                                                 width=1)))
    
    upper_bound = go.Scatter(name='Upper Confidence Interval',
                             x=dates,
                             showlegend=False,
                             xaxis='x1',
                             yaxis='y1',
                             y=df['3s'],
                             marker=dict(color='#444'),
                             line=dict(color=('rgb(23, 96, 167)'),
                                       width=2,
                                       dash='dash'),
                             fillcolor='rgba(68, 68, 68, 0.3)',
                             fill='tonexty')
    
    lower_bound = go.Scatter(name='Confidence Interval',
                             x=dates,
                             xaxis='x1',
                             yaxis='y1',
                             y=df['-3s'],
                             marker=dict(color='#444'),
                             line=dict(color=('rgb(23, 96, 167)'),
                                       width=2,
                                       dash='dash'),
                             fillcolor='rgba(68, 68, 68, 0.3)',
                             fill='tonexty')
    
    sales = go.Scatter(name='Sales',
                       x=dates,
                       y=df['sales'],
                       xaxis='x2',
                       yaxis='y2',
                       mode='lines',
                       marker=dict(size=12,
                                   line=dict(width=1),
                                   color='blue'))
    
    predicted = go.Scatter(name='Predicted',
                           x=dates,
                           y=df['predicted'],
                           xaxis='x2',
                           yaxis='y2',
                           mode='lines',
                           marker=dict(size=12,
                                       line=dict(width=1),
                                       color='silver'))
    
    anomalies_map_level_1 = go.Scatter(name='Anomaly Sales Level 1',
                               showlegend=False,
                               x=dates,
                               y=anomaly_points_level_1,
                               mode='markers',
                               xaxis='x2',
                               yaxis='y2',
                               marker=dict(color='yellow',
                                           size=11,
                                           line=dict(color='yellow',
                                                     width=1)))
    
    anomalies_map_level_2 = go.Scatter(name='Anomaly Sales Level 2',
                               showlegend=False,
                               x=dates,
                               y=anomaly_points_level_2,
                               mode='markers',
                               xaxis='x2',
                               yaxis='y2',
                               marker=dict(color='orange',
                                           size=11,
                                           line=dict(color='orange',
                                                     width=1)))
    
    anomalies_map_level_3 = go.Scatter(name='Anomaly Sales Level 3',
                               showlegend=False,
                               x=dates,
                               y=anomaly_points_level_3,
                               mode='markers',
                               xaxis='x2',
                               yaxis='y2',
                               marker=dict(color='red',
                                           size=11,
                                           line=dict(color='red',
                                                     width=1)))
    
    axis = dict(showline=True,
                zeroline=False,
                showgrid=True,
                mirror=True,
                ticklen=4,
                gridcolor='#ffffff',
                tickfont=dict(size=10))
    
    layout = dict(width=950,
                  height=950,
                  autosize=False,
                  title=metric_name,
                  margin=dict(l=0, r=0, t=50, b=10),
                  showlegend=True,
                  legend=dict(font=dict(size=10)),
                  xaxis1=dict(axis,
                              **dict(domain=[0, 1],
                                     anchor='y1',
                                     showticklabels=True)),
                  xaxis2=dict(axis,
                              **dict(domain=[0, 1],
                                     anchor='y2',
                                     showticklabels=True)),
                  yaxis1=dict(axis,
                              **dict(domain=[0.70, 1],
                                     anchor='x1',
                                     hoverformat='.2f')),
                  yaxis2=dict(axis,
                              **dict(domain=[0.34, 0.64],
                                     anchor='x2',
                                     hoverformat='.2f')))
    
    fig = go.Figure(data=[table, upper_bound, lower_bound, sales,
                          predicted, mvingavrg, error,
                          anomalies_level_1, anomalies_level_2, anomalies_level_3,
                          anomalies_map_level_1, anomalies_map_level_2, anomalies_map_level_3],
                    layout=layout)
    
    return iplot(fig)


# In[164]:


plot_anomaly(classify_df, "Daily Sales Anomaly Detection")


# Al utilizar una media móvil y una desviación estándar aquí, se evitan falsas anomalías continuas durante escenarios como los días de grandes ventas. Se resalta el primer pico o caída, después de lo cual se ajustan los umbrales. Además, la tabla que proporciona datos reales predijo el cambio y el formato condicional en función del nivel de anomalías.

# Por último, se propone una alternativa para detectar anomalías, en este caso por distintos niveles de porcentaje de error entre los valores reales y los predichos.

# In[165]:


def detect_classify_anomalies_percentages(df, window):
    df = df.copy()
    df.replace([np.inf, -np.inf], np.NaN, inplace=True)
    df.fillna(0, inplace=True)
    df['error'] = abs(df['sales'] - df['predicted'])
    df['percentage_change'] = ((df['error']) / df['sales']) * 100
    df['25p_error'] = (df['sales'] * 0.25)
    df['50p_error'] = (df['sales'] * 0.5)
    df['75p_error'] = (df['sales'] * 0.75)
    df['100p_error'] = (df['sales'] * 1)
    cut_list = df[['error', '25p_error', '50p_error', '75p_error', '100p_error']]
    cut_values = cut_list.values
    cut_sort = np.sort(cut_values)
    df['impact'] = [(lambda x: np.where(cut_sort == df['error'].iloc[x])[1][0])(x) for x in range(len(df['error']))]
    severity = {0:0, 1:1, 2:2, 3:3, 4:4}
    df['color'] = df['impact'].map(severity)
    df['anomaly_points_level_1'] = np.where(df['color'] == 1, df['error'], np.nan)
    df['anomaly_points_level_2'] = np.where(df['color'] == 2, df['error'], np.nan)
    df['anomaly_points_level_3'] = np.where(df['color'] == 3, df['error'], np.nan)
    df['anomaly_points_level_4'] = np.where(df['color'] == 4, df['error'], np.nan)
    df = df.sort_values(by='date', ascending=False)
    df.date = pd.to_datetime(df['date'].astype(str), format='%Y-%m-%d')
    
    return df


# In[166]:


classify_df = detect_classify_anomalies_percentages(predicted_df, 7)


# In[167]:


classify_df.reset_index(inplace=True)


# In[168]:


classify_df = classify_df.drop("index", axis=1)


# In[169]:


classify_df.date = classify_df.date.dt.strftime('%Y-%m-%d')


# In[170]:


classify_df


# In[171]:


def plot_anomaly_percentages(df, metric_name):
    dates = df.date
    
    bool_array_level_1 = (abs(df['anomaly_points_level_1']) > 0)
    sales_level_1 = df["sales"][-len(bool_array_level_1):]
    anomaly_points_level_1 = bool_array_level_1 * sales_level_1
    anomaly_points_level_1[anomaly_points_level_1 == 0] = np.nan
    
    bool_array_level_2 = (abs(df['anomaly_points_level_2']) > 0)
    sales_level_2 = df["sales"][-len(bool_array_level_2):]
    anomaly_points_level_2 = bool_array_level_2 * sales_level_2
    anomaly_points_level_2[anomaly_points_level_2 == 0] = np.nan
    
    bool_array_level_3 = (abs(df['anomaly_points_level_3']) > 0)
    sales_level_3 = df["sales"][-len(bool_array_level_3):]
    anomaly_points_level_3 = bool_array_level_3 * sales_level_3
    anomaly_points_level_3[anomaly_points_level_3 == 0] = np.nan
    
    bool_array_level_4 = (abs(df['anomaly_points_level_4']) > 0)
    sales_level_4 = df["sales"][-len(bool_array_level_4):]
    anomaly_points_level_4 = bool_array_level_4 * sales_level_4
    anomaly_points_level_4[anomaly_points_level_4 == 0] = np.nan
    
    color_map = {0:'rgba(228, 222, 249, 0.65)', 1:'yellow', 2:'orange', 3:'red', 4:'darkred'}
    
    table = go.Table(
        domain = dict(x=[0, 1],
                      y=[0, 0.45]),
        columnwidth = [1, 2],
        header = dict(height=20,
                      values=[['<b>Date</b>'], ['<b>Sales</b>'],
                              ['<b>Predicted</b>'], ['<b>% Difference</b>'], ['<b>Severity (0-3)</b>']],
                      font=dict(color=['rgb(45, 45, 45)'] * 5, size=14),
                      fill=dict(color='#d562be')),
        cells = dict(values=[df.round(2)[k].tolist() for k in ['date', 'sales', 'predicted',
                                                               'percentage_change', 'color']],
                     line=dict(color='#506784'),
                     align=['center'] * 5,
                     font=dict(color=['rgb(40, 40, 40)'] * 5, size=12),
                     suffix=[None] + [''] + [''] + ['%'] + [''],
                     height=27,
                     fill=dict(color=[df['color'].map(color_map)])))
    
    sales = go.Scatter(name='Sales',
                       x=dates,
                       y=df['sales'],
                       xaxis='x1',
                       yaxis='y1',
                       mode='lines',
                       marker=dict(size=12,
                                   line=dict(width=1),
                                   color='blue'))
    
    predicted = go.Scatter(name='Predicted',
                           x=dates,
                           y=df['predicted'],
                           xaxis='x1',
                           yaxis='y1',
                           mode='lines',
                           marker=dict(size=12,
                                       line=dict(width=1),
                                       color='silver'))
    
    anomalies_map_level_1 = go.Scatter(name='Anomaly Sales Level 1',
                               x=dates,
                               y=anomaly_points_level_1,
                               mode='markers',
                               xaxis='x1',
                               yaxis='y1',
                               marker=dict(color='yellow',
                                           size=11,
                                           line=dict(color='yellow',
                                                     width=1)))
    
    anomalies_map_level_2 = go.Scatter(name='Anomaly Sales Level 2',
                               x=dates,
                               y=anomaly_points_level_2,
                               mode='markers',
                               xaxis='x1',
                               yaxis='y1',
                               marker=dict(color='orange',
                                           size=11,
                                           line=dict(color='orange',
                                                     width=1)))
    
    anomalies_map_level_3 = go.Scatter(name='Anomaly Sales Level 3',
                               x=dates,
                               y=anomaly_points_level_3,
                               mode='markers',
                               xaxis='x1',
                               yaxis='y1',
                               marker=dict(color='red',
                                           size=11,
                                           line=dict(color='red',
                                                     width=1)))
    
    anomalies_map_level_4 = go.Scatter(name='Anomaly Sales Level 4',
                               x=dates,
                               y=anomaly_points_level_4,
                               mode='markers',
                               xaxis='x1',
                               yaxis='y1',
                               marker=dict(color='darkred',
                                           size=11,
                                           line=dict(color='darkred',
                                                     width=1)))
    
    axis = dict(showline=True,
                zeroline=False,
                showgrid=True,
                mirror=True,
                ticklen=4,
                gridcolor='#ffffff',
                tickfont=dict(size=6))
    
    layout = dict(width=950,
                  height=950,
                  autosize=False,
                  title=metric_name,
                  margin=dict(l=0, r=0, t=50, b=10),
                  showlegend=True,
                  legend=dict(yanchor="top",
                              y=0.99,
                              xanchor="left",
                              x=0.01,
                              font=dict(size=10)),
                  xaxis1=dict(axis,
                              **dict(domain=[0, 1],
                                     anchor='y1',
                                     showticklabels=True)),
                  yaxis1=dict(axis,
                              **dict(domain=[0.5, 1],
                                     anchor='x1',
                                     hoverformat='.2f')))
    
    fig = go.Figure(data=[table, sales, predicted,
                          anomalies_map_level_1, anomalies_map_level_2, anomalies_map_level_3, anomalies_map_level_4],
                    layout=layout)
    
    return iplot(fig)


# In[172]:


plot_anomaly_percentages(classify_df, "Daily Sales Anomaly Detection (by percentages)")

