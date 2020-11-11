import urllib.request as re
import pymysql

def readMidiPage():
    url_str = 'https://www.midiworld.com/composers.htm'
    response = re.urlopen(url_str)
    page_str = response.read().decode('gbk')
    open('midiPage.html', 'w').write(page_str)

def readBaiduTop():
    url_str = 'http://top.baidu.com/buzz?b=341&c=513&fr=topbuzz_b1_c513'
    response = re.urlopen(url_str)
    page_str = response.read().decode('utf-8')
    lines = page_str.strip().split('\r\n')
    topNewsLink = []
    titles = []
    for l in lines:
        if 'list-title' in l:
            topNewsLink.append(l)
    for tn in topNewsLink:
        tn = tn.replace('</a>', '')
        title = tn[tn.rindex('>')+1:]
        titles.append(title)
    titles = titles[:100]
    for ti in titles:
        print(ti)

def readMySQL():
    db = pymysql.connect('localhost', 'root', '900327', 'mydatabase', charset='utf8')
    cursor = db.cursor()
    sql = 'select name, team, position from nbaplayers'
    try:
        cursor.execute(sql)
        res = cursor.fetchall()
        print(res)
    except:
        print('select failed.')
    db.close()

if __name__ == "__main__":
    readMidiPage()
    