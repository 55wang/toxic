from datetime import datetime
from flask import render_template, flash, redirect, url_for, request, \
    send_from_directory, jsonify, Markup
from flask_login import login_user, logout_user, current_user, login_required
from werkzeug.urls import url_parse
from app import app, db
from app.forms import LoginForm, RegistrationForm, EditProfileForm, PostForm, \
    ResetPasswordRequestForm, ResetPasswordForm, TwitterForm
from app.models import User, Post, Tweet
from app.email import send_password_reset_email
from code.twitter_crawl import twitter_query
from code.LDA_topic_modeling import LDA_model
from code.LSI_topic_modeling import LSI_model
import numpy as np
from code.utils import *
import os
from collections import Counter
import pandas as pd
import pickle
from nltk.corpus import stopwords


def test_transform(test, tf_transformer, NB_model, RF_model, KNN_model, SVM_model, GB_model):
    pkl_file = open('./data/lowTF_words.pkl', 'rb')
    lowTF_words = pickle.load(pkl_file)

    porter = nltk.PorterStemmer()
    stops = set(stopwords.words('english'))
    stops.add('rt')
    test['comment_text'] = test['comment_text'].apply(lambda x: x.replace('\n', ' '))

    tweets_new = []
    for index, tweet in test.iterrows():
        words = tweet['comment_text'].split(' ')
        new = []
        for w in words:
            if w not in lowTF_words:
                new.append(w)
        new_tweet = ' '.join(new)
        tweets_new.append(new_tweet)
    test_feats = tf_transformer.transform(tweets_new)
    # test_predicts = model.predict(test_feats)
    result_df = pd.DataFrame()
    NB_predicts = NB_model.predict(test_feats)
    result_df['NB_predicts'] = NB_predicts
    RF_predicts = RF_model.predict(test_feats)
    result_df['RF_predicts'] = RF_predicts
    KNN_predicts = KNN_model.predict(test_feats)
    result_df['KNN_predicts'] = KNN_predicts
    SVM_predicts = SVM_model.predict(test_feats)
    result_df['SVM_predicts'] = SVM_predicts
    GB_predicts = GB_model.predict(test_feats)
    result_df['GB_predicts'] = GB_predicts
    result_df['majority_vote'] = result_df.mode(axis=1)[0]
    # print result_df.head()
    predicts = result_df['majority_vote'].tolist()

    # print(predicts[0], predicts, int(predicts[0]))

    test_predicts = int(predicts[0])
    if test_predicts == 1:
        prediction = 'Toxic'
    elif test_predicts == 0:
        prediction = 'Normal'
    # print 'test_predicts: ', prediction
    return prediction


@app.before_request
def before_request():
    if current_user.is_authenticated:
        current_user.last_seen = datetime.utcnow()
        db.session.commit()


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
@login_required
def index():
    form = PostForm()
    if form.validate_on_submit():
        print('posting data: ' + form.post.data)

        with open(os.path.join('./data', 'model_transformer.pkl')) as output:
            tf_transformer, NB_model, RF_model, KNN_model, SVM_model, GB_model = pickle.load(output)

        test = pd.DataFrame({'id': [123], 'comment_text': [form.post.data]})
        prediction = test_transform(test, tf_transformer, NB_model, RF_model, KNN_model, SVM_model, GB_model)

        post = Post(body=form.post.data, author=current_user, prediction=prediction)
        db.session.add(post)
        db.session.commit()
        flash('Your post is now live!')
        return redirect(url_for('index'))
    page = request.args.get('page', 1, type=int)
    posts = current_user.followed_posts().paginate(
        page, app.config['POSTS_PER_PAGE'], False)
    next_url = url_for('explore', page=posts.next_num) \
        if posts.has_next else None
    prev_url = url_for('explore', page=posts.prev_num) \
        if posts.has_prev else None
    return render_template('index.html', title='Home', form=form,
                           posts=posts.items, next_url=next_url,
                           prev_url=prev_url)


@app.route('/explore')
@login_required
def explore():
    page = request.args.get('page', 1, type=int)
    posts = Post.query.order_by(Post.timestamp.desc()).paginate(
        page, app.config['POSTS_PER_PAGE'], False)
    next_url = url_for('explore', page=posts.next_num) \
        if posts.has_next else None
    prev_url = url_for('explore', page=posts.prev_num) \
        if posts.has_prev else None
    return render_template('index.html', title='Explore', posts=posts.items,
                           next_url=next_url, prev_url=prev_url)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user is None or not user.check_password(form.password.data):
            flash('Invalid username or password')
            return redirect(url_for('login'))
        login_user(user, remember=form.remember_me.data)
        next_page = request.args.get('next')
        if not next_page or url_parse(next_page).netloc != '':
            next_page = url_for('index')
        return redirect(next_page)
    return render_template('login.html', title='Sign In', form=form)


@app.route('/tweets', methods=['GET', 'POST'])
def tweets():
    form = TwitterForm()
    if form.validate_on_submit():
        print('posting data: ' + form.post.data)
        tweets = twitter_query('USA', form.post.data)
        for tweet in tweets:

            with open(os.path.join('./data', 'model_transformer.pkl')) as output:
                tf_transformer, NB_model, RF_model, KNN_model, SVM_model, GB_model = pickle.load(output)

            test = pd.DataFrame({'id': [123], 'comment_text': [tweet]})
            prediction = test_transform(test, tf_transformer, NB_model, RF_model, KNN_model, SVM_model, GB_model)

            post = Tweet(keyword=form.post.data, body=tweet, prediction=prediction)
            db.session.add(post)
        db.session.commit()
        flash('Twitter crawled!')
        return redirect(url_for('tweets'))
    posts = Tweet.query.all()
    return render_template('tweets.html', title='tweets', form=form,
                           posts=posts)


@app.route('/stats')
def stats():
    posts = Tweet.query.all()
    post_list = []
    for post in posts:
        post_list.append(post.body)
    norm_corpus = normalize_corpus(post_list, only_text_chars=True, tokenize=True)

    flattened_norm_corpus = [y for x in norm_corpus for y in x]

    vectorizer, feature_matrix = build_feature_matrix(flattened_norm_corpus, feature_type='frequency')
    occ = np.asarray(feature_matrix.sum(axis=0)).ravel().tolist()
    counts_df = pd.DataFrame({'term': vectorizer.get_feature_names(), 'occurrences': occ})
    counts_df = counts_df.sort_values(by='occurrences', ascending=False).head(15).to_dict('records')

    # print counts_df
    count_result = []
    for pair in counts_df:
        temp = {}
        temp['label'] = pair['term'].encode('ascii', 'ignore')
        temp['value'] = pair['occurrences']
        count_result.append(temp)

    count_result = Markup(count_result)
    # print result

    transformer = TfidfVectorizer(analyzer='word')
    norm_corpus = normalize_corpus(post_list, only_text_chars=True, tokenize=False)

    transformed_weights = transformer.fit_transform(norm_corpus)
    weights = np.asarray(transformed_weights.mean(axis=0)).ravel().tolist()
    weights_df = pd.DataFrame({'term': transformer.get_feature_names(), 'weight': weights})
    weights_df = weights_df.sort_values(by='weight', ascending=False).head(15).to_dict('records')

    tfidf_result = []
    for pair in weights_df:
        temp = {}
        temp['label'] = pair['term'].encode('ascii', 'ignore')
        temp['value'] = pair['weight']
        tfidf_result.append(temp)

    tfidf_result = Markup(tfidf_result)
    return render_template('stats.html', title='Statistics', count_result=count_result, tfidf_result=tfidf_result)


@app.route('/lda')
def lda():
    posts = Tweet.query.all()
    post_list = []
    for post in posts:
        post_list.append(post.body)

    vectorizer, lda_model, svd_transformer, svd_matrix = LDA_model(post_list, 3, 100)

    data = {}
    feat_names = vectorizer.get_feature_names()
    for compNum in range(len(lda_model.components_)):
        print compNum
        comp = lda_model.components_[compNum]

        # Sort the weights in the first component, and get the indices
        indices = np.argsort(comp).tolist()[::-1]

        # Grab the top 10 terms which have the highest weight in this component.
        terms = [feat_names[weightIndex] for weightIndex in indices[0:10]]
        weights = [comp[weightIndex] for weightIndex in indices[0:10]]
        # print terms, weights
        # terms.reverse()
        # weights.reverse()
        # print terms, weights
        result = {}
        result['terms'] = terms
        result['weights'] = weights
        data[compNum] = result
    print data

    topic1 = []
    topic2 = []
    topic3 = []

    for first_key in data.iterkeys():
        print first_key
        if first_key == 0:
            for term, weight in zip(data[first_key]['terms'], data[first_key]['weights']):
                temp = {}
                temp['label'] = term.encode('ascii', 'ignore')
                temp['value'] = weight
                topic1.append(temp)
        elif first_key == 1:
            for term, weight in zip(data[first_key]['terms'], data[first_key]['weights']):
                temp = {}
                temp['label'] = term.encode('ascii', 'ignore')
                temp['value'] = weight
                topic2.append(temp)
        elif first_key == 2:
            for term, weight in zip(data[first_key]['terms'], data[first_key]['weights']):
                temp = {}
                temp['label'] = term.encode('ascii', 'ignore')
                temp['value'] = weight
                topic3.append(temp)

    print topic1
    print topic2
    print topic3

    topic1 = Markup(topic1)
    topic2 = Markup(topic2)
    topic3 = Markup(topic3)
    # print chart_data
    return render_template('lda.html', title='LDA', topic1=topic1, topic2=topic2, topic3=topic3)


@app.route('/lsi')
def lsi():
    posts = Tweet.query.all()
    post_list = []
    for post in posts:
        post_list.append(post.body)

    vectorizer, lda_model, svd_transformer, svd_matrix = LSI_model(post_list, 3, 100)

    data = {}
    feat_names = vectorizer.get_feature_names()
    for compNum in range(len(lda_model.components_)):
        print compNum
        comp = lda_model.components_[compNum]

        # Sort the weights in the first component, and get the indices
        indices = np.argsort(comp).tolist()[::-1]

        # Grab the top 10 terms which have the highest weight in this component.
        terms = [feat_names[weightIndex] for weightIndex in indices[0:10]]
        weights = [comp[weightIndex] for weightIndex in indices[0:10]]
        # print terms, weights
        # terms.reverse()
        # weights.reverse()
        # print terms, weights
        result = {}
        result['terms'] = terms
        result['weights'] = weights
        data[compNum] = result
    print data

    topic1 = []
    topic2 = []
    topic3 = []

    for first_key in data.iterkeys():
        print first_key
        if first_key == 0:
            for term, weight in zip(data[first_key]['terms'], data[first_key]['weights']):
                temp = {}
                temp['label'] = term.encode('ascii', 'ignore')
                temp['value'] = weight
                topic1.append(temp)
        elif first_key == 1:
            for term, weight in zip(data[first_key]['terms'], data[first_key]['weights']):
                temp = {}
                temp['label'] = term.encode('ascii', 'ignore')
                temp['value'] = weight
                topic2.append(temp)
        elif first_key == 2:
            for term, weight in zip(data[first_key]['terms'], data[first_key]['weights']):
                temp = {}
                temp['label'] = term.encode('ascii', 'ignore')
                temp['value'] = weight
                topic3.append(temp)

    print topic1
    print topic2
    print topic3

    topic1 = Markup(topic1)
    topic2 = Markup(topic2)
    topic3 = Markup(topic3)
    # print chart_data
    return render_template('lsi.html', title='LSI', topic1=topic1, topic2=topic2, topic3=topic3)


@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash('Congratulations, you are now a registered user!')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)


@app.route('/reset_password_request', methods=['GET', 'POST'])
def reset_password_request():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = ResetPasswordRequestForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user:
            send_password_reset_email(user)
        flash('Check your email for the instructions to reset your password')
        return redirect(url_for('login'))
    return render_template('reset_password_request.html',
                           title='Reset Password', form=form)


@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    user = User.verify_reset_password_token(token)
    if not user:
        return redirect(url_for('index'))
    form = ResetPasswordForm()
    if form.validate_on_submit():
        user.set_password(form.password.data)
        db.session.commit()
        flash('Your password has been reset.')
        return redirect(url_for('login'))
    return render_template('reset_password.html', form=form)


@app.route('/user/<username>')
@login_required
def user(username):
    user = User.query.filter_by(username=username).first_or_404()
    page = request.args.get('page', 1, type=int)
    posts = user.posts.order_by(Post.timestamp.desc()).paginate(
        page, app.config['POSTS_PER_PAGE'], False)
    next_url = url_for('user', username=user.username, page=posts.next_num) \
        if posts.has_next else None
    prev_url = url_for('user', username=user.username, page=posts.prev_num) \
        if posts.has_prev else None
    return render_template('user.html', user=user, posts=posts.items,
                           next_url=next_url, prev_url=prev_url)


@app.route('/edit_profile', methods=['GET', 'POST'])
@login_required
def edit_profile():
    form = EditProfileForm(current_user.username)
    if form.validate_on_submit():
        current_user.username = form.username.data
        current_user.about_me = form.about_me.data
        db.session.commit()
        flash('Your changes have been saved.')
        return redirect(url_for('edit_profile'))
    elif request.method == 'GET':
        form.username.data = current_user.username
        form.about_me.data = current_user.about_me
    return render_template('edit_profile.html', title='Edit Profile',
                           form=form)


@app.route('/follow/<username>')
@login_required
def follow(username):
    user = User.query.filter_by(username=username).first()
    if user is None:
        flash('User {} not found.'.format(username))
        return redirect(url_for('index'))
    if user == current_user:
        flash('You cannot follow yourself!')
        return redirect(url_for('user', username=username))
    current_user.follow(user)
    db.session.commit()
    flash('You are following {}!'.format(username))
    return redirect(url_for('user', username=username))


@app.route('/unfollow/<username>')
@login_required
def unfollow(username):
    user = User.query.filter_by(username=username).first()
    if user is None:
        flash('User {} not found.'.format(username))
        return redirect(url_for('index'))
    if user == current_user:
        flash('You cannot unfollow yourself!')
        return redirect(url_for('user', username=username))
    current_user.unfollow(user)
    db.session.commit()
    flash('You are not following {}.'.format(username))
    return redirect(url_for('user', username=username))


root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'web')


@app.route('/<path:path>', methods=['GET'])
def static_proxy(path):
    print root
    return send_from_directory(root, path)