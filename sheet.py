import pandas as pd
import json
import openpyxl


class Sheet():
    """This class reads from/writes on the google-dsheet 
    which we have created before analyzing the video file 
    by using of it's api.
    It contains multiple sheets:
    -Players: General information of the players such as ID, First name, Last name, Gender, Year
    -Match: General information of the matches we have analyzed such as Round, Gender, 
    -Plays
    -Serves
    -Positions
    -Plaus-help
    -Shots
    -Shots-help
    """
    
    def __init__(self, sheets):
        self.sheets = sheets
        self.N_games = None
        self.N_shots = None

    def read_playsheet(self):
        Plays_sheet = pd.read_excel(self.sheets, sheet_name="Plays")
        df_plays = Plays_sheet[Plays_sheet['Start frame'] != '']   # Discarding blank rows
        #df_plays.pop(df_plays.columns[0])   # Discarding of first column
        self.N_games = len(df_plays.index)
        return df_plays, self.N_games

    def read_matchsheet(self):
        match_sheet = pd.read_excel(self.sheets, sheet_name="Match")
        match_sheet = match_sheet[match_sheet['Name'] != '']
        df_match = pd.DataFrame(match_sheet, columns=match_sheet.columns)
        return df_match
    
    def read_possheet(self):
        pos_sheet = pd.read_excel(self.sheets, sheet_name="Positions")
        pos_sheet = pos_sheet[pos_sheet['Frame'] != '']
        return pos_sheet
    
    def read_shotsheet(self):
        Shots_sheet = pd.read_excel(self.sheets, sheet_name="Shots")
        df_shots = Shots_sheet[Shots_sheet['Frame'] != '']   # Discarding blank rows
        df_shots.pop(df_shots.columns[0])   # Discarding of first column
        self.N_shots = len(df_shots.index)
        return df_shots, self.N_shots
 
    def WritePos2Sheet(self, D_Positions):
        print(f'D dictionary is about to dump')
        df_pos = pd.DataFrame(D_Positions, columns=D_Positions.keys())
        #df_pos.to_pickle('df_pos.pkl')
        pos_sheet = self.read_possheet()
        df = pd.DataFrame(pos_sheet, columns=pos_sheet.columns)
        #df.to_pickle('df.pkl')
        if not df_pos['Frame'].isin(df['Frame']).all():   # To avoid overwriting inside "Positions" sheet
            df_updated = df.append(df_pos)
            df_updated.to_pickle('df_updated.pkl')
            with pd.ExcelWriter(self.sheets, mode="a", if_sheet_exists="replace", engine="openpyxl") as writer:
                pd.DataFrame(df_updated).to_excel(writer, sheet_name='Positions', index=False)
        else: pass

    def ShotChecker(self, frame_no, shot_no):
        who_is_hitter = None
        df_shots, N_shots = self.read_shotsheet()
        shot_ident = df_shots[df_shots['Frame']==frame_no]
        if not shot_ident.empty:
            who_is_hitter = shot_ident['Who'][0]
            shot_no += 1
        return who_is_hitter, shot_no
######################################################

class PointsID():
    def __init__(self, sp_sheets):
        sheet_ins = Sheet(sp_sheets) # Build an instance from Sheet class
        self.sheets = sheet_ins.sheets # call "sheets" attribute
        self.plays_sheet,_ = sheet_ins.read_playsheet() # call "read_playsheet" method from instance
        self.match_sheet = sheet_ins.read_matchsheet() # call "read_matchsheet" method from instance

        self.point_score = ['0', '0']
        self.game_score = ['0', '0']
        self.set_score = ['0', '0']
        self.G_counter = 0  # extra game (due to golden game)
        self.S_counter = 0  # extra set (due to golden set)

    def ResetPoints(self):
        self.point_score = ['0', '0']
    def ResetGames(self):
        self.game_score = ['0', '0']
    def ResetSets(self):
        self.set_score = ['0', '0']

    def gender_labeler(self, game_no):
        if game_no <= 103:
            self.plays_sheet.loc[game_no, "Gender"] = "Feminine"
        else:
            self.plays_sheet.loc[game_no, "Gender"] = "Masculine"

    def incrementpoint(self, game_no):
        POINTS = ('0', '15', '30', '40')
        top_team = self.plays_sheet['Top Team'][game_no]
        winner = self.plays_sheet['Winner (T, B, N)'][game_no]

        if winner == 'T':
            if top_team == 'A':
                scorer = 'A'
            elif top_team == 'B':
                scorer = 'B'
        elif winner == 'B':
            if top_team == 'A':
                scorer = 'B'
            elif top_team == 'B':
                scorer = 'A'
        elif winner == 'N':
            scorer = 'N'

        self.plays_sheet.loc[game_no, "Scored games A"] = self.game_score[0]
        self.plays_sheet.loc[game_no, "Scored games B"] = self.game_score[1]
        self.plays_sheet.loc[game_no, "Scored points A"] = self.point_score[0]
        self.plays_sheet.loc[game_no, "Scored points B"] = self.point_score[1]

        if self.game_score[0] == self.game_score[1] == str(5 + self.G_counter):  # Golden game
            self.G_counter += 1
        if self.set_score[0] == self.set_score[1] == str(1 + self.S_counter):  # Golden set
            self.S_counter += 1

        if scorer == 'N':
                self.plays_sheet.loc[game_no,"Scored points A"] = self.point_score[0]
                self.plays_sheet.loc[game_no,"Scored points B"] = self.point_score[1]
        else:
            if (self.point_score[0] == '40') & (scorer == 'A'):
                self.point_score[0] = 'WIN GAME'
                self.plays_sheet.loc[game_no, "Scored points A"] = self.point_score[0]
                self.ResetPoints()
                assert (self.plays_sheet.loc[game_no, "End_game"] == '*')
                if self.game_score[0] == str(5 + self.G_counter):  # End of the set
                    self.game_score[0] = 'WIN SET'
                    self.plays_sheet.loc[game_no, "Scored games A"] = self.game_score[0]
                    self.ResetGames()
                    self.G_counter = 0
                    assert (self.plays_sheet.loc[game_no, "End_set"] == '*')
                    self.set_score[0] = str(int(self.set_score[0]) + 1)
                    if self.set_score[0] == str(2 + self.S_counter):  # End of the set
                        self.ResetSets()
                        self.S_counter = 0
                        assert (self.plays_sheet.loc[game_no, "End_match"] == '*')
                else:  # Set is not finished yet
                    self.game_score[0] = str(int(self.game_score[0]) + 1)  # add a point to set points
                    self.plays_sheet.loc[game_no, "Scored games A"] = self.game_score[0] # write to df
            elif (self.point_score[1] == '40') & (scorer == 'B'):
                self.point_score[1] = 'WIN GAME'
                self.plays_sheet.loc[game_no, "Scored points B"] = self.point_score[1]
                self.ResetPoints()
                assert (self.plays_sheet.loc[game_no, "End_game"] == '*')
                if self.game_score[1] == str(5 + self.G_counter):  # End of the set
                    self.game_score[1] = 'WIN SET'
                    self.plays_sheet.loc[game_no, "Scored games B"] = self.game_score[1]
                    self.set_score[0] = str(int(self.set_score[0]) + 1)
                    self.ResetGames()
                    self.G_counter = 0
                    assert (self.plays_sheet.loc[game_no, "End_set"] == '*')
                    if self.set_score[1] == str(2 + self.S_counter):  # End of the set
                        self.ResetSets()
                        self.S_counter = 0
                        assert (self.plays_sheet.loc[game_no, "End_match"] == '*')
                else:  # Set is not finished yet
                    self.game_score[1] = str(int(self.game_score[1]) + 1) # add a point to set points
                    self.plays_sheet.loc[game_no, "Scored games B"] = self.game_score[1] # write to df
            else:
                if scorer == 'A':
                    self.point_score[0] = POINTS[POINTS.index(self.point_score[0]) + 1]
                    self.plays_sheet.loc[game_no, "Scored points A"] = self.point_score[0]
                elif scorer == 'B':
                    self.point_score[1] = POINTS[POINTS.index(self.point_score[1]) + 1]
                    self.plays_sheet.loc[game_no, "Scored points B"] = self.point_score[1]
        return self.plays_sheet


    def PlayerID(self, game_no):
        top_team = self.plays_sheet['Top Team'][game_no]
        AL_F = self.match_sheet.loc[0, "Team A Left side player"]
        AR_F = self.match_sheet.loc[0, "Team A Right side player"]
        BL_F = self.match_sheet.loc[0, "Team B Left side player"]
        BR_F = self.match_sheet.loc[0, "Team B Right side player"]

        AL_M = self.match_sheet.loc[1, "Team A Left side player"]
        AR_M = self.match_sheet.loc[1, "Team A Right side player"]
        BL_M = self.match_sheet.loc[1, "Team B Left side player"]
        BR_M = self.match_sheet.loc[1, "Team B Right side player"]

        if self.plays_sheet.loc[game_no, "Gender"] == "Feminine":
            if top_team == 'A':
                self.plays_sheet.loc[game_no, "Who_is_TL"] = AL_F
                self.plays_sheet.loc[game_no, "Who_is_TR"] = AR_F
                self.plays_sheet.loc[game_no, "Who_is_BL"] = BL_F
                self.plays_sheet.loc[game_no, "Who_is_BR"] = BR_F
            elif top_team == 'B':
                self.plays_sheet.loc[game_no, "Who_is_TL"] = BR_F
                self.plays_sheet.loc[game_no, "Who_is_TR"] = BL_F
                self.plays_sheet.loc[game_no, "Who_is_BL"] = AR_F
                self.plays_sheet.loc[game_no, "Who_is_BR"] = AL_F
        elif self.plays_sheet.loc[game_no, "Gender"] == "Masculine":
            if top_team == 'A':
                self.plays_sheet.loc[game_no, "Who_is_TL"] = AL_M
                self.plays_sheet.loc[game_no, "Who_is_TR"] = AR_M
                self.plays_sheet.loc[game_no, "Who_is_BL"] = BL_M
                self.plays_sheet.loc[game_no, "Who_is_BR"] = BR_M
            elif top_team == 'B':
                self.plays_sheet.loc[game_no, "Who_is_TL"] = BR_M
                self.plays_sheet.loc[game_no, "Who_is_TR"] = BL_M
                self.plays_sheet.loc[game_no, "Who_is_BL"] = AR_M
                self.plays_sheet.loc[game_no, "Who_is_BR"] = AL_M
        return self.plays_sheet


    def PointsID2PlaySheet(self, game_no):
        self.gender_labeler(game_no)
        self.incrementpoint(game_no)
        self.PlayerID(game_no)

        with pd.ExcelWriter(self.sheets, mode="a", if_sheet_exists="replace", engine="openpyxl") as writer:
            self.plays_sheet.to_excel(writer, sheet_name='Plays', index=False)

