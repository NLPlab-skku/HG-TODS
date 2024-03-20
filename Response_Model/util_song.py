import json, os
import dgl
import torch
from bartenc import bartenc_conceptnet
from tabulate import tabulate

class sg_utils():
    def __init__(self, title_map, person_map, title_list, person_list, personal, tokenizer):

        self.title_map = title_map
        self.person_map = person_map
        self.title_list = title_list
        self.person_list = person_list
        self.personal = personal
        self.tokenizer = tokenizer 

    def _songs_to_graph(self, songs):     
            
            title_map = self.title_map
            person_map = self.person_map 

            edge_map = {
                'artist' : 0,
                'artist_inv' : 1,
                'writer' : 2,
                'writer_inv' : 3,
                'composer' : 4,
                'composer_inv' : 5
            }

            # edge : [[node1, node2], [node1, node2]] 형태로 구성
            e_artist, e_artist_inverse = [], []
            e_writer, e_writer_inverse = [], []
            e_composer, e_composer_inverse = [], []
            
            for song in songs:
                title, artist, writer, composer = song['title'], song['artist'], song['writer'], song['composer']

                for a in artist:
                    if a == '':
                        continue
                    e_artist.append([title_map[title], person_map[a]])
                    e_artist_inverse.append([person_map[a], title_map[title]])

                for w in writer:
                    if w == '':
                        continue
                    e_writer.append([title_map[title], person_map[w]])
                    e_writer_inverse.append([person_map[w], title_map[title]])

                for c in composer:
                    if c == '':
                        continue
                    e_composer.append([title_map[title], person_map[c]])
                    e_composer_inverse.append([person_map[c], title_map[title]])
                    
            data_dict = {
                ('title', 'artist', 'person') : e_artist,
                ('person', 'artist_inv', 'title') : e_artist_inverse,
                ('title', 'writer', 'person') : e_writer,
                ('person', 'writer_inv', 'title') : e_writer_inverse,
                ('title', 'composer', 'person') : e_composer,
                ('person', 'composer_inv', 'title') : e_composer_inverse,
                ('dummy', 'dummy', 'dummy') : [[0, 0]]
            }
            
            num_nodes_dict = {
                'title' : len(title_map),
                'person' : len(person_map),
                'dummy' : 1
            }

            g = dgl.heterograph(data_dict, num_nodes_dict = num_nodes_dict)
            g = dgl.to_simple(g)

            table = [
                {
                    '# Title Nodes' : g.num_nodes('title'),
                    '# Person Nodes' : g.num_nodes('person'),
                }
            ]
            print(tabulate(table, headers = 'keys'))

            g.ndata['label'] = {
                'title' : torch.tensor([i for i in range(len(title_map))]),
                'person' : torch.tensor([i for i in range(len(person_map))]),
                'dummy' : torch.tensor([0])
            }

            # initialize node embedding
            bartenc_embedding = torch.load(f"/workspace/NRF/cache/bartenc_embedding")

            assert g.num_nodes('title') == bartenc_embedding['title_embedding'].shape[0]
            assert g.num_nodes('person') == bartenc_embedding['person_embedding'].shape[0]

            g.ndata['emb'] = {
                'title' : bartenc_embedding['title_embedding'],
                'person' : bartenc_embedding['person_embedding'],
                'dummy' : torch.zeros((1, bartenc_embedding['title_embedding'].shape[1]), dtype = bartenc_embedding['title_embedding'].dtype)
            }

            relation_embedding = {
                'artist' : bartenc_embedding['relation_embedding'][0, :],
                'artist_inv' : bartenc_embedding['relation_embedding'][1, :],
                'writer' : bartenc_embedding['relation_embedding'][2, :],
                'writer_inv' : bartenc_embedding['relation_embedding'][3, :],
                'composer' : bartenc_embedding['relation_embedding'][4, :],
                'composer_inv' : bartenc_embedding['relation_embedding'][5, :]
            }

            g.edata['emb'] = {
                ('title', 'artist', 'person') : relation_embedding['artist'].unsqueeze(0).expand(g.num_edges(('title', 'artist', 'person')), -1),
                ('person', 'artist_inv', 'title') : relation_embedding['artist_inv'].unsqueeze(0).expand(g.num_edges(('person', 'artist_inv', 'title')), -1),
                ('title', 'writer', 'person') : relation_embedding['writer'].unsqueeze(0).expand(g.num_edges(('title', 'writer', 'person')), -1),
                ('person', 'writer_inv', 'title') : relation_embedding['writer_inv'].unsqueeze(0).expand(g.num_edges(('person', 'writer_inv', 'title')), -1),
                ('title', 'composer', 'person') : relation_embedding['composer'].unsqueeze(0).expand(g.num_edges(('title', 'composer', 'person')), -1),
                ('person', 'composer_inv', 'title') : relation_embedding['composer_inv'].unsqueeze(0).expand(g.num_edges(('person', 'composer_inv', 'title')), -1),
                ('dummy', 'dummy', 'dummy') : torch.zeros((1, bartenc_embedding['title_embedding'].shape[1]), dtype = bartenc_embedding['title_embedding'].dtype)
            }

            self.songs_graph = g

            return g


    def _slot_to_subgraph(self, user_id, prev_slot):

        '''
            "user_slots": {
                    "song title": "Ransomware (Feat. Moldy)",
                    "singer": "",
                    "genre": "",
                    "composer": "",
                    "lyricist": "",
                    "playlist type": "",
                    "playlist title": "",
                    "recommendation": "",
                    "topic": "",
                    "concept": ""
                }
        '''


        '''
            slot1 : title -
            slot2 : person -
            slot3 : genre -
            slot4 : person -
            slot5 : person -
            slot7 : (recent_play_list, custom_play_list, favorite_songs) -
            slot8 : title of custom_play_list -
            slot10 : favorite song -
            slot11 : favorite artist *
            slot12 : favorite genre *
            slot13 : favorite composer *
            slot14 : favorite writer *
        '''

        title_map, person_map = self.title_map, self.person_map
        user_info = self.personal[user_id]

        title, person, genre = [], [], []

        # Not use personal info
        if prev_slot['song title'] != '':
            title += prev_slot['song title'].split('//')
        if prev_slot['singer'] != '':
            person += prev_slot['singer'].split('//')
        if prev_slot['genre'] != '':
            genre += prev_slot['genre'].split('//')
        if prev_slot['composer'] != '':
            person += prev_slot['composer'].split('//')
        if prev_slot['lyricist'] != '':
            person += prev_slot['lyricist'].split('//')

        # Use personal info : playlist
        # To Do : for loop
        try:
            if prev_slot['playlist type'] == 'recent_play_list':
                for s in user_info['recent_playlist']:
                    title.append(s['title'])
            if prev_slot['playlist type'] == 'custom_playlist' and prev_slot['playlist title'] != '':
                playlist_title = prev_slot['playlist title']
                if not playlist_title in user_info['playlist']:
                    playlist_title = " " + playlist_title
                if not playlist_title in user_info['playlist']:
                    playlist_title = playlist_title.strip() + " "
                if '독백' not in playlist_title and 'Folk' not in playlist_title and '이어폰' not in playlist_title and playlist_title in user_info['playlist']: 
                    for s in user_info['playlist'][playlist_title]:
                        title.append(s['title'])
                if playlist_title not in user_info['playlist']:
                    for playlist_songlist in user_info['playlist'].values():
                        for song in list(playlist_songlist):
                            title.append(song['title'])
            elif prev_slot['playlist type'] == 'custom_playlist' and prev_slot['playlist title'] == '':
                for playlist_title, song_list in user_info['playlist'].items():
                    for s in song_list:
                        title.append(s['title'])
        except:
            print(playlist_title)
            print(user_info['playlist'])

        # Use personal info : favorite
        if prev_slot['recommendation'] == True:
            title += user_info['recommendation']

        title, person = list(set(title)), list(set(person))
        title = [title_map[name] for name in title if name in title_map]
        person = [person_map[name] for name in person if name in person_map]

        if title == [] and person == []:
            sg, _ = dgl.khop_out_subgraph(self.songs_graph, {'dummy' : [0]}, k = 1)
            return sg

        else:
            sg, _ = dgl.khop_out_subgraph(self.songs_graph, {'title' : title, 'person' : person}, k = 2)

            return sg
        

    def _sg2triples(self, sg):
        edges = {
            'artist' : '가수',
            'writer' : '작사가',
            'composer' : '작곡가',
        }
        triples = []
        for e in sg.etypes:
            src, dst = sg.edges(etype=e)
            if e in edges:
                src_temp = sg.ndata['label']['title'][src]
                dst_temp = sg.ndata['label']['person'][dst]
                for s, d in zip(src_temp, dst_temp):
                    title = self.title_list[int(s)]
                    person = self.person_list[int(d)]

                    triples.append(f"{title};{edges[e]};{person}")

            elif e == 'dummy':
                continue

            else:
                src_temp = sg.ndata['label']['person'][src]
                dst_temp = sg.ndata['label']['title'][dst]
                for s, d in zip(src_temp, dst_temp):
                    person = self.person_list[int(s)]
                    title = self.title_list[int(d)]

                    triples.append(f"{person};제목;{title}")

        output = f"{self.tokenizer.sep_token}".join(triples)

        return output
