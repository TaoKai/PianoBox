import urllib.request as re
import pymysql
from codecs import open
import os, sys, shutil
import requests

def readMidiPage():
    url_str = 'https://www.midiworld.com/composers.htm'
    response = re.urlopen(url_str)
    page_str = response.read().decode('utf-8')
    lines = page_str.split('\n')
    links = {}
    for l in lines:
        if '<LI><A HREF=' in l:
            startl = l.index('\"')+1
            l = l[startl:]
            endl = l.index('\"')
            link = 'https://www.midiworld.com/'+l[:endl]
            l = l[endl:]
            startn = l.index('>')+1
            l = l[startn:].strip()
            endn = l.index('<')-1
            name = l[:endn]
            links[name]={'link': link}
    for k, v in links.items():
        link = v['link']
        response = re.urlopen(link)
        page_str = response.read().decode('utf-8')
        print(k, link)
        lines = page_str.split('\n')
        save_dir = 'midi_classics/'+k.replace(' ', '')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        else:
            continue
        try:
            for l in lines:
                if '.mid\"' in l:
                    try:
                        startl = l.index('\"')
                        l = l[startl+1:]
                        endl = l.index('\"')
                        link = l[:endl]
                        l = l[endl+1:]
                        startn = l.index('>')+1
                        l = l[startn:]
                        endn = l.index('>')
                        name = l[:endn-3].strip()
                        save_path = save_dir+'/'+name+'.mid'
                        save_path = save_path.replace(' ', '').replace('\"', '').replace('..', '.')
                        if os.path.exists(save_path):
                            save_path = save_path.split('.')[0]+'I.mid'
                        r = requests.get(link)
                        if r.content is not None and len(r.content)>800:
                            open(save_path, 'wb').write(r.content)
                            print(save_path)
                    except Exception as e:
                        print(str(e))
                        continue
        except:
            continue

            

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

def file_proc():
    path = 'midi_classics'
    dirs = [path+'/'+d for d in os.listdir(path)]
    for d in dirs:
        for fn in os.listdir(d):
            p = d+'/'+fn
            if '.mid' not in fn:
                print(p)
                os.remove(p)

if __name__ == "__main__":
    file_proc()
    