import requests

qq_number = input('请输入QQ号: ')
api_url = f'https://zy.xywlapi.cc/qqcx?qq={qq_number}'
infos = requests.get(api_url).json()
print(f'''通过{qq_number}号查询到的个人信息如下:
密保手机号: {infos.get("phone")},
号码归属地: {infos.get("phonediqu")},
lol信息: {infos.get("lol")},
微博UID: {infos.get("wb")},
''')
