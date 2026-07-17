CREATE TABLE IF NOT EXISTS films (
    film_id INT PRIMARY KEY,              
    title TEXT NOT NULL,
    genres JSONB,                         
    keywords JSONB,                       
    overview TEXT,                        
    release_date DATE,                   
    vote_average NUMERIC(3, 1) DEFAULT 0.0
);