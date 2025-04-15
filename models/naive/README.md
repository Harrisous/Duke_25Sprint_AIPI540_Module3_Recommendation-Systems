## Naive Approach
Design steps:
1. With any user with the movie ratings, we first find the most similar existing user by cosine similarity.
2. We compare the movie that the existing user has watched but the new user hasn't. Sort those not seen/rated movies' rating from high to low and pick the top-k to the new user.