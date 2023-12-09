# %%
import googlemaps
import pandas as pd
import requests
import time
import numpy as np
# %% 
api_key = "APIKEY"
# %%
client = googlemaps.Client(api_key)
# %%
lonRange = [136.828765, 136.929999]  # the range of longitude 
latRange = [35.1425, 35.191666]  # the range of latitude   
lonDivision = 0.001 # メッシュ
latDivision = 0.001 #  メッシュ
rad = 100
outfile = "output1.csv"

pt_range = pd.read_csv(r"E:\50_DATABASE\boundary_poi_0001.csv")
# %%
# places = []
# while True:
#     response = requests.get('https://maps.googleapis.com/maps/api/place/nearbysearch/json', params=query)
#     result = response.json()
#     places.extend(result['results'])
#     if 'next_page_token' not in result:
#         break
#     next_page_token = result['next_page_token']
#     query['pagetoken'] = next_page_token

# # %%
# dic = []
# for lon in np.arange(lonRange[0], lonRange[1], lonDivision):
#     for lat in np.arange(latRange[0], latRange[1], latDivision):
#         loc = {'lat':lat, 'lng':lon}
#         print(loc)
#         nexttoken = ""
#         while nexttoken != 'None':
#             time.sleep(3)
#             # print("#########################################")
#             # print(nexttoken)
#             place_result = client.places_nearby(location=loc, radius=rad, page_token= nexttoken)
#             nexttoken = place_result["next_page_token"] if "next_page_token" in place_result else 'None'
#             for item in place_result["results"]:
#                 data = {}
#                 if "business_status" in item:
#                     data["lat"] = item["geometry"]["location"]["lat"]
#                     data["lng"] = item["geometry"]["location"]["lng"]
#                     data["place_id"] = item["place_id"]
#                     data["name"] = item["name"]
#                     data["types"] = item["types"]
#                 dic.append(data)
# %%
pt_range


# %%
dic = []
for pt in pt_range.values:
    print(pt[0])
    loc = {'lat':pt[1], 'lng':pt[2]}
    nexttoken = ""
    while nexttoken != 'None':
        time.sleep(3)
        # print("#########################################")
        # print(nexttoken)
        place_result = client.places_nearby(location=loc, radius=rad, page_token= nexttoken)
        nexttoken = place_result["next_page_token"] if "next_page_token" in place_result else 'None'
        for item in place_result["results"]:
            data = {}
            if "business_status" in item:
                data["lat"] = item["geometry"]["location"]["lat"]
                data["lng"] = item["geometry"]["location"]["lng"]
                data["place_id"] = item["place_id"]
                data["name"] = item["name"]
                data["types"] = item["types"]

                if item['types']:
                    for index,d in enumerate(item['types']):
                        data['type'+str(index)]=d
                else:
                    data['type'] =None
            dic.append(data)
# %%
df = pd.DataFrame.from_dict(dic)
df.drop_duplicates(subset='place_id').to_csv("google_poi.csv")

# %%
df

# %%
df.to_csv("google_poi.csv")

# %%
