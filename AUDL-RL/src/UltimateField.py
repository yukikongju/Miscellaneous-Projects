import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsRegressor
from audl.stats.endpoints.gamestats import GameStats
from audl.stats.static.utils import get_throw_type


class FieldSquare(object):

    def __init__(self, throw_name: str, throw_side: str, 
                 surface_proba_mean: float, surface_proba_var: float, 
                 x_delta_mean: float, x_delta_var: float,
                 y_delta_mean: float, y_delta_var: float):
        self.throw_name = throw_name
        self.throw_side = throw_side
        self.surface_proba_mean = surface_proba_mean
        self.surface_proba_var = surface_proba_var
        self.x_delta_mean = x_delta_mean
        self.x_delta_var = x_delta_var
        self.y_delta_mean = y_delta_mean
        self.y_delta_var = y_delta_var


class UltimateFieldThrowDistribution(object):

    def __init__(self, df_throws: pd.DataFrame):
        self.df_throws = df_throws

        self.OFFSET_WIDTH = 27
        self.FIELD_WIDTH = 55
        self.FIELD_LENGTH = 120
        self.NUM_NEIGHBORS = 2 # make sure not greater than min throws

        self._init_throw_distributions()


    def _init_throw_distributions(self):
        """

        """
        self.throws_distribution = {} # key=(throw_type, throw_side) ; value=FieldSquare
        throws_types = ['dump', 'swing', 'dish', 'huck', 'pass']
        throws_sides = ['left', 'right']
        for t_type, t_side in itertools.product(throws_types, throws_sides):
            self.throws_distribution[(t_type, t_side)] = self.__init_throw_side_distribution(throw_type=t_type, throw_side=t_side)

    def __init_throw_side_distribution(self, throw_type: str, throw_side: str): # FIXME: check why X has NAN value
        """

        """
        # --- filter out rows
        is_throw_type = self.df_throws['throw_type'] == throw_type
        is_throw_side = self.df_throws['throw_side'] == throw_side
        df = self.df_throws[is_throw_type & is_throw_side]

        # --- initialize KNNs to predict (1) proba (2) x_delta (3) y_delta
        points = np.array(df[['x_field', 'y_field']])
        successes = np.array(df[['successful']])
        x_deltas, y_deltas = np.array(df[['x']]), np.array(df[['y']])

        knn_proba = KNeighborsRegressor(n_neighbors = self.NUM_NEIGHBORS, 
                                        weights = 'distance', algorithm='auto',
                                        metric='minkowski')
        knn_proba.fit(points, successes)
        
        knn_x = KNeighborsRegressor(n_neighbors = self.NUM_NEIGHBORS, 
                                    weights = 'distance', algorithm='auto', 
                                    metric='minkowski')
        knn_x.fit(points, x_deltas)

        knn_y = KNeighborsRegressor(n_neighbors = self.NUM_NEIGHBORS, 
                                    weights = 'distance', algorithm='auto', 
                                    metric='minkowski')
        knn_y.fit(points, y_deltas)

        def get_knn_prediction(current_pos: np.array, 
                               knn: KNeighborsRegressor, Y) -> FieldSquare:
            v_mean = knn.predict([current_pos])
            distances, indices = knn.kneighbors([current_pos])
            neighbors_vals = Y[indices]
            v_variance = np.var(neighbors_vals)
            return v_mean, v_variance


        # --- compute throw distribution for all squares using KNN
        field = []
        for x_pos in range(self.FIELD_WIDTH):
            row = []
            for y_pos in range(self.FIELD_LENGTH):
                # init current position
                current_pos = np.array([x_pos-self.OFFSET_WIDTH, y_pos])

                # compute mean and variance for: proba, x_delta, y_delta
                proba_mean, proba_var = get_knn_prediction(current_pos, knn_proba, successes)
                x_delta_mean, x_delta_var = get_knn_prediction(current_pos, knn_x, x_deltas)
                y_delta_mean, y_delta_var = get_knn_prediction(current_pos, knn_y, y_deltas)

                # init FieldSquare
                square = FieldSquare(throw_name=throw_type, throw_side=throw_side, 
                                     surface_proba_mean=proba_mean, surface_proba_var=proba_var,
                                     x_delta_mean=x_delta_mean, x_delta_var=x_delta_var,
                                     y_delta_mean=y_delta_mean, y_delta_var=y_delta_var)
                row.append(square)
            field.append(row)

        return field


    def render(self): # TODO
        """
        Render field with matplotlib

        """
        pass


class UltimateGameResults(object):

    """
    Class which computes throwing results of a given game for a specific team
    """

    def __init__(self, game_id: str, team_ext_id: str):
        """
        Parameters
        ---------
        game_id: str
            > ex: "2023-08-26-SLC-NY"
        team_ext_id: str
            > ex: "empire". Must be one of the team that plays that game
        """
        self.game_id = game_id
        self.team_ext_id = team_ext_id
        self.game = GameStats(self.game_id)

        self.df_throws = self._get_throws_dataframe()
        self.throws_distributions = UltimateFieldThrowDistribution(self.df_throws)


    def _get_throws_dataframe(self):
        """
        Function that computes dataframe of throws and cleans it up

        DataFrame:
        - Keys: game_id, point, thrower_id, receiver_id, throw_type, throw_side, 
          thrower_ext_id, receiver_ext_id, thrower_full_name, receiver_full_name
        - Cols: throw_distance, x, y, x_field, y_field, angle_degrees, successful
        """
        # fetch throws dataset from API
        df = self.game.get_throws_dataframe()

        # filter out required team
        df = df[df['team_ext_id'] == self.team_ext_id]

        # split df into valid throws and throwaways
        valid_throws_types = ['dump', 'swing', 'dish', 'huck', 'pass']
        is_valid_throw = df['throw_type'].isin(valid_throws_types)
        df_valid_throws = df[is_valid_throw]
        df_valid_throws.loc[:, 'successful'] = True

        df_throwaways = df[~is_valid_throw]
        df_throwaways.loc[:, 'throw_type'] = df_throwaways.apply(lambda r: get_throw_type(x1=0, y1=0, x2=r['x'], y2=r['y'])[0], axis=1)
        df_throwaways.loc[:, 'throw_side'] = df_throwaways.apply(lambda r: get_throw_type(x1=0, y1=0, x2=r['x'], y2=r['y'])[1], axis=1)
        df_throwaways.loc[:, 'successful'] = False

        # concatenate back
        df = pd.concat([df_valid_throws, df_throwaways]).reset_index(drop=True)
        return df

    def get_throws_rate_dataframe(self):
        """
        Function that return dataframe with completion rate and selection rate 
        for each dataframe

        DataFrame:
        - Keys: 
        - Cols:
          * completion_rate:
          * selection_rate:

        """
        # calculate total successful throws per team
        df_throws_counts = self.df_throws.groupby(['team_ext_id']).agg({'successful': ['sum', 'size']}).reset_index()
        df_throws_counts = df_throws_counts.rename(columns={'successful': 'total'})

        # calculate overall selection probability
        df_selection = self.df_throws.groupby(['team_ext_id', 'throw_type']).agg({'successful': ['sum', 'size']}).reset_index()

        df_selection = df_selection.merge(df_throws_counts, on=['team_ext_id']).reset_index()
        df_selection['completion_rate'] = df_selection['successful']['sum'] / df_selection['successful']['size']
        df_selection['selection_rate'] = df_selection['successful']['size'] / df_selection['total']['size']

        return df_selection

    def get_throws_surface_rate_dataframe(self):
        """
        Function that compute surface probability success

        DataFrame:
        - Keys: 
        - Cols: 
          * completion_rate:
          * selection_type:
          * selection_rate:
        """
        df_throws_counts = self.df_throws.groupby(['team_ext_id']).agg({'successful': ['sum', 'size']}).reset_index()
        df_throws_counts = df_throws_counts.rename(columns={'successful': 'total'})

        df_surface = self.df_throws.groupby(['team_ext_id', 'throw_type', 'throw_side']).agg({'successful': ['sum', 'size']}).reset_index()

        df_throws_type_counts = self.df_throws.groupby(['team_ext_id', 'throw_type']).agg({'successful': ['sum', 'size']}).reset_index()
        df_throws_type_counts = df_throws_type_counts.rename(columns={'successful': 'total_type'})
        df_surface = df_surface.merge(df_throws_type_counts, on=['team_ext_id', 'throw_type'])
        df_surface = df_surface.merge(df_throws_counts, on=['team_ext_id'])

        df_surface['completion_rate'] = df_surface['successful']['sum'] / df_surface['successful']['size']
        df_surface['selection_type'] = df_surface['successful']['size'] / df_surface['total_type']['size']

        df_surface['selection_rate'] = df_surface['successful']['size'] / df_surface['total']['size']

        return df_surface       

    def plot_throw_scatterplot(self, throw_type: str):
        """
        Plot throw's success failure with a scatterplot
        """ 
        is_throw_type = self.df_throws['throw_type'] == throw_type
        df_pass = self.df_throws[is_throw_type]
        
        colors = df_pass['successful'].map({True: 'green', False: 'red'})
        plt.scatter(x=df_pass['x_field'], y=df_pass['y_field'], c=colors, alpha=0.5)
        plt.title(f"Scatterplot of {self.team_ext_id} {throw_type} attempts")
        plt.show()


#  --------------------------------------------------------------------

def main():
    game = UltimateGameResults(game_id="2023-08-26-SLC-NY", 
                                       team_ext_id="empire")
    #  print(game.df_throws.columns)
    #  print(game.get_throws_rate_dataframe())
    #  game.plot_throw_scatterplot(throw_type='swing')
    print(game.throws_distributions)

    

if __name__ == "__main__":
    main()
