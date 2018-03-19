import tweepy
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener

class Init_Connection():
    def Init_Connection(self):
        consumer_key = '4N18JRcCs4a6XsqBybCuBQjcI'
        consumer_secret = 'W1vbhCYjVSYbuFaaFvpp4Qsg5JLndROn62aONu2VzvoT6tzYVA'
        access_token = '973754658840875008-nBhYaqxNips6HTkXNgIKC0HkMiyicnV'
        access_secret = 'MV1BxlnasCel8jrgz5SPMlSgkUGyyONimxc2qZDRgDJiq'

        auth = OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_secret)

        api = tweepy.API(auth)
        return api, auth


class Stream_Listener(StreamListener):
    def on_data(self, data):
        try:
            f = open('Output.json', 'a')
            f.write(data)
            print(data)
            return True
        except BaseException as e:
            print("Error = " + str(e))
        return True
    def on_error(self, status):
        print "Error"
        print(status)
        return True

if __name__ == '__main__':
    init =  Init_Connection()
    api, auth = init.Init_Connection()
    t_stream = Stream(auth, Stream_Listener())
    t_stream.filter(track=['#F1', '#Halo'])
