import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
import json 
from tensorflow.keras.applications.inception_v3 import preprocess_input

st.title("🍴Food Classification🍴")

tab1, tab2 = st.tabs([" Let's Predict", "About"])

with tab1:
    model_path = hf_hub_download(
        repo_id="erlangram/food101_model",
        filename="food_model.keras")

    model = tf.keras.models.load_model(model_path, compile=False)
    st.write("Input shape model:", model.input_shape)


    def preprocess_image(uploaded_file, target_size=(299, 299)):
        image = Image.open(uploaded_file).convert("RGB")
        image = image.resize(target_size)
        img_array = np.array(image, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array, image

    #Urutan label sesuai indeks
    with open("class_indices.json", "r") as f:
        class_indices = json.load(f)

    food_labels = [None] * len(class_indices)

    for label, idx in class_indices.items():
        food_labels[idx] = label

    uploaded = st.file_uploader("📤 Upload or drag an image...", type=["jpg", "jpeg", "png", "webp"])

    if uploaded is not None:
        ori_image = Image.open(uploaded).convert("RGB")
        img_array, image = preprocess_image(uploaded)
        st.image(ori_image, caption="Uploaded Image", use_container_width=True)

        pred = model.predict(img_array)
        top5_idx = pred[0].argsort()[-5:][::-1]


        predicted_food = food_labels[top5_idx[0]]
        st.subheader(f"Prediction: **{predicted_food}**")

        def get_food_description(food_name):
            descriptions = {
            "Apple Pie": "A classic dessert strongly associated with American cuisine, though its origins trace back to Europe. It consists of a flaky pastry crust filled with spiced apples (often mixed with cinnamon, sugar, and sometimes nutmeg or lemon juice). The filling becomes soft and sweet after baking, while the crust remains golden and crisp. Apple pie is commonly served warm, sometimes topped with whipped cream, cheddar cheese, or a scoop of vanilla ice cream",
            "Baby Back Ribs": "A cut of pork ribs taken from the upper portion of the ribcage, near the backbone. They are smaller, leaner, and more tender compared to spare ribs, which is why they are called “baby” back ribs. This dish is especially popular in American barbecue culture, often seasoned with dry rubs or slow-cooked and then glazed with smoky-sweet barbecue sauce. When properly prepared, the meat becomes tender enough to fall off the bone",
            "Baklava": "A rich and sweet pastry that originates from the Middle East and the Ottoman Empire, still widely enjoyed in Turkey, Greece, and neighboring countries. It is made by layering sheets of thin phyllo dough with finely chopped nuts (commonly pistachios, walnuts, or almonds), then baking and soaking the pastry in honey or sugar syrup. The result is a sticky, crunchy, and aromatic dessert that balances sweetness with nutty flavors.",
            "Beef Carpaccio": "An Italian appetizer created in Venice in the mid-20th century. It consists of raw beef tenderloin or sirloin sliced extremely thin (almost translucent) and arranged flat on a plate. It is usually dressed with olive oil, lemon juice, capers, arugula, and shaved Parmesan cheese. The dish emphasizes freshness and delicate flavors, highlighting the natural taste of high-quality beef.",
            "Beef Tartare": "A dish found across Europe, especially France and Belgium, made from finely chopped or minced raw beef. The meat is typically mixed with seasonings such as onions, capers, mustard, Worcestershire sauce, and herbs. A raw egg yolk is often placed on top before serving. Unlike carpaccio, tartare is minced rather than sliced, giving it a different texture. It is usually eaten with toasted bread or crackers.",
            "Beet Salad": "A fresh and colorful salad featuring beets as the main ingredient. The beets can be roasted, boiled, or pickled, and are often paired with complementary ingredients like goat cheese, walnuts, citrus fruits, or leafy greens. The earthy sweetness of the beets contrasts well with tangy cheeses and crunchy nuts. This salad is considered nutritious and is popular in both European and modern Western cuisines.",
            "Beignets": "A deep-fried pastry of French origin, famously popular in New Orleans, Louisiana. They are small square pieces of dough, fried until puffy and golden, then heavily dusted with powdered sugar. Beignets are typically served hot, often alongside a cup of café au lait (coffee with chicory and milk). Their airy texture and sweet coating make them a beloved breakfast or snack food.",
            "Bibimbap": "A signature Korean dish whose name literally means “mixed rice.” It consists of a bowl of warm rice topped with an assortment of vegetables (such as spinach, bean sprouts, mushrooms, and zucchini), sliced meat (commonly beef bulgogi), and a fried or raw egg. It is served with gochujang (a spicy red chili paste), which is mixed into the dish before eating. The combination of flavors and textures—sweet, savory, spicy, crunchy, and soft—makes bibimbap one of the most iconic Korean meals.",
            "Bread Pudding": "A comforting dessert with origins in Europe, traditionally made as a way to use up stale bread. The bread is soaked in a mixture of milk or cream, eggs, sugar, and spices such as cinnamon or nutmeg, then baked until set. The result is a soft, custard-like interior with a slightly crisp top. Variations may include raisins, nuts, or a sauce such as vanilla cream, whiskey sauce, or caramel.",
            "Breakfast Burrito": "A hearty Tex-Mex style dish consisting of a flour tortilla wrapped around typical breakfast foods. The filling usually includes scrambled eggs, potatoes, cheese, and breakfast meats such as bacon, sausage, or chorizo. Salsa, beans, or avocado may also be added. Portable and filling, the breakfast burrito is a popular grab-and-go meal in the southwestern United States and beyond.",
            "Bruschetta": "An Italian appetizer made with slices of toasted bread rubbed with garlic and topped with olive oil, salt, and various toppings. The most classic version features fresh tomatoes, basil, and mozzarella or Parmesan cheese. It’s a light and refreshing starter, often enjoyed with wine.",
            "Caesar Salad": "A salad invented in Mexico by Italian-American restaurateur Caesar Cardini. It typically includes romaine lettuce, croutons, Parmesan cheese, and Caesar dressing (a creamy mixture made with anchovies, garlic, lemon juice, egg yolk, mustard, and Worcestershire sauce). Often topped with grilled chicken or shrimp.",
            "Cannoli": "A traditional Sicilian dessert consisting of tube-shaped fried pastry shells filled with sweet ricotta cheese, often flavored with vanilla, chocolate chips, candied fruits, or pistachios. They are crispy on the outside and creamy inside.",
            "Caprese Salad": "A simple and fresh Italian salad named after the island of Capri. It consists of sliced fresh mozzarella, tomatoes, and basil, usually drizzled with olive oil and sometimes balsamic vinegar. Its colors (red, white, green) reflect the Italian flag.",
            "Carrot Cake": "A moist and spiced cake made with grated carrots, which add natural sweetness and texture. Often flavored with cinnamon and nutmeg, and topped with cream cheese frosting. Sometimes includes walnuts, raisins, or pineapple.",
            "Ceviche": "A dish popular in Latin America, especially Peru. It consists of raw fish or seafood “cooked” by marinating in citrus juice (commonly lime or lemon). The acidity denatures the proteins, giving the fish a firm texture. It’s usually mixed with onions, chili peppers, cilantro, and corn or sweet potato on the side.",
            "Cheese Plate": "An assortment of cheeses, often served with crackers, bread, fruit, nuts, and jams. It can include soft cheeses (like Brie or Camembert), hard cheeses (like Cheddar or Manchego), and blue cheeses. A popular choice for appetizers or wine pairings.",
            "Cheesecake": "A rich dessert made with a creamy filling of cream cheese, sugar, and eggs, set on a crust made of crushed cookies or graham crackers. New York-style cheesecake is dense and smooth, while other versions may be lighter or baked with ricotta. Often topped with fruit, chocolate, or caramel.",
            "Chicken Curry": "A broad category of dishes made with chicken simmered in a sauce rich with spices. Indian versions often use onions, tomatoes, garlic, ginger, and spices like turmeric, cumin, and coriander. Other variations include Thai chicken curry with coconut milk and curry paste.",
            "Chicken Quesadilla": "A Mexican dish consisting of a tortilla filled with cheese and chicken, folded in half, and grilled or pan-fried until the cheese melts. Often served with salsa, sour cream, or guacamole.",
            "Chicken Wings": "Deep-fried or baked chicken wings, often coated in sauces. The Buffalo wing (from Buffalo, New York) is famous for its spicy hot sauce and butter coating. Other variations include barbecue, honey garlic, or teriyaki wings.",
            "Chocolate Cake": "A universally loved dessert made with cocoa or melted chocolate. It ranges from light sponge cakes to dense, fudgy variations, and can be layered with frosting or ganache.",
            "Chocolate Mousse": "A light, airy dessert made by folding whipped cream or beaten egg whites into a rich chocolate mixture. Originating in France, it has a smooth, fluffy texture and can be topped with whipped cream or berries.",
            "Churros": "A Spanish and Latin American fried-dough pastry, typically sprinkled with sugar and cinnamon. Often served with thick hot chocolate or dulce de leche for dipping. Crunchy outside, soft inside.",
            "Clam Chowder": "A creamy soup popular in New England (USA), made with clams, potatoes, onions, and celery. Another version, Manhattan clam chowder, is tomato-based instead of creamy.",
            "Club Sandwich": "A layered sandwich with toasted bread, typically filled with turkey or chicken, bacon, lettuce, tomato, and mayonnaise. Often cut into quarters and held with toothpicks.",
            "Crab Cakes": "A seafood dish from the eastern United States, especially Maryland. They are patties made from crab meat mixed with breadcrumbs, mayonnaise, and seasonings, then pan-fried or baked until golden.",
            "Creme Brulee": "A French dessert consisting of a rich vanilla custard base topped with a thin layer of caramelized sugar. The top is torched until it becomes a crisp shell that cracks with a spoon.",
            "Croque Madame": "A French sandwich made with ham, cheese, and béchamel sauce, grilled to golden perfection. When topped with a fried egg, it’s called “Croque Madame”; without the egg, it’s “Croque Monsieur.”",
            "Cup Cakes": "Small individual cakes baked in paper cups, often frosted and decorated. They come in many flavors and are popular for parties and celebrations.",
            "Deviled Eggs": "Hard-boiled eggs cut in half, with the yolks mashed and mixed with mayonnaise, mustard, and spices, then piped back into the whites. Popular as a party appetizer.",
            "Donuts": "Sweet fried dough rings or filled pastries, often glazed, powdered, or covered in chocolate and sprinkles. Variations exist worldwide, like jelly-filled donuts or cream-filled ones.",
            "Dumplings": "A broad category of dough-filled or dough-wrapped foods found in many cultures. Chinese dumplings (jiaozi) are often filled with pork, shrimp, or vegetables. They can be boiled, steamed, or fried. Other versions exist in Eastern Europe, Japan, and beyond.",
            "Edamame": "Young, green soybeans served in their pods, lightly boiled or steamed and sprinkled with salt. Popular as a Japanese appetizer or snack.",
            "Eggs Benedict": "A brunch favorite from the U.S., consisting of poached eggs placed on toasted English muffins with Canadian bacon or ham, all topped with hollandaise sauce (a buttery lemon-egg yolk sauce).",
            "Escargots": "A French delicacy of cooked snails, typically served in their shells with garlic, butter, and parsley. Considered a luxurious appetizer.",
            "Falafel": "A Middle Eastern dish made of deep-fried balls or patties of ground chickpeas or fava beans mixed with herbs and spices. Usually served in pita bread with vegetables and tahini sauce.",
            "Filet Mignon": "A highly prized cut of beef taken from the smaller end of the tenderloin. Known for being extremely tender with a delicate flavor. Often pan-seared or grilled and served as a steak.",
            "Fish and Chips": "A British classic of battered and deep-fried fish (commonly cod or haddock) served with thick-cut fried potatoes (“chips”). Often accompanied by tartar sauce, malt vinegar, or mushy peas.",
            "Foie Gras": "A French delicacy made from the fattened liver of a duck or goose. It has a rich, buttery texture and is often served as a pâté or seared with accompaniments like fruit compote.",
            "French Fries": "Deep-fried potato strips, golden and crispy on the outside, soft on the inside. Served worldwide as a side dish, snack, or fast food, often with ketchup, mayonnaise, or other dips.",
            "French Onion Soup": "A savory soup made with caramelized onions simmered in beef broth, topped with a slice of bread and melted cheese (usually Gruyère). A comforting French classic.",
            "French Toast": "Slices of bread soaked in a mixture of eggs and milk, then fried until golden. Often topped with syrup, powdered sugar, or fruit. Known in France as pain perdu (“lost bread”), originally made to use stale bread.",
            "Fried Calamari": "Squid rings coated in batter or breadcrumbs and deep-fried until crisp. Usually served as an appetizer with marinara sauce, lemon wedges, or aioli.",
            "Fried Rice": "A versatile Asian dish made by stir-frying cooked rice with ingredients such as vegetables, eggs, soy sauce, and often meat or seafood. A common way to use leftover rice.",
            "Frozen Yogurt": "A frozen dessert similar to ice cream but made with yogurt instead of cream. It has a tangier flavor and is often considered a lighter, healthier option. Popularly served in cups with toppings such as fruit, chocolate, or candy.",
            "Garlic Bread": "Slices of bread (often baguette or Italian bread) topped with garlic butter and herbs, then baked or grilled. It’s usually served as a side dish with pasta, soups, or salads.",
            "Gnocchi": "Small, soft dumplings from Italy, usually made with potatoes, flour, and egg. They are boiled and then served with sauces like tomato, pesto, or butter and sage. Known for their pillowy texture.",
            "Greek Salad": "A Mediterranean salad made with cucumbers, tomatoes, red onions, olives, and feta cheese, dressed with olive oil and oregano. Refreshing, colorful, and healthy.",
            "Grilled Cheese Sandwich": "An American comfort food consisting of bread grilled with butter and filled with melted cheese inside. Often served with tomato soup.",
            "Grilled Salmon": "Salmon fillets seasoned and grilled, resulting in a smoky, tender dish. Often paired with lemon, herbs, and vegetables for a nutritious main course.",
            "Guacamole": "A creamy Mexican dip made from mashed avocados mixed with lime juice, onions, tomatoes, cilantro, and chili peppers. Usually served with tortilla chips.",
            "Gyoza": "Japanese dumplings typically filled with ground meat (often pork) and vegetables. They are pan-fried on one side for crispiness and steamed to cook the filling. Adapted from Chinese jiaozi.",
            "Hamburger": "A sandwich consisting of one or more cooked patties of ground meat, usually beef, placed inside a sliced bread roll or bun. Often garnished with lettuce, tomato, onion, pickles, cheese, and condiments like ketchup and mustard.",
            "Hot and Sour Soup": "A Chinese soup with a spicy and tangy flavor, made from ingredients like mushrooms, tofu, bamboo shoots, and egg ribbons, flavored with soy sauce, vinegar, and white pepper.",
            "Hot Dog": "A grilled or steamed sausage (usually pork, beef, or chicken) placed in a sliced bun, topped with mustard, ketchup, onions, relish, or sauerkraut. A popular American street food.",
            "Huevos Rancheros": "A traditional Mexican breakfast dish of fried eggs served on corn tortillas, topped with tomato-chili sauce, beans, rice, and avocado. Hearty and flavorful.",
            "Hummus": "A Middle Eastern dip made from blended chickpeas, tahini (sesame paste), olive oil, lemon juice, and garlic. Creamy, nutty, and healthy, usually eaten with pita bread or vegetables.",
            "Ice Cream": "A frozen dessert made from cream, sugar, and flavorings (like chocolate, vanilla, or fruit). It comes in countless flavors and textures, served in cones, cups, or as part of sundaes.",
            "Lasagna": "An Italian baked pasta dish with layers of pasta sheets, meat sauce (ragù), béchamel or ricotta, and cheese, baked until golden and bubbling. A hearty and comforting main course.",
            "Lobster Bisque": "A rich, creamy French soup made from lobster stock, cream, butter, and often sherry or cognac. Known for its smooth texture and luxurious flavor.",
            "Lobster Roll Sandwich": "A New England specialty: lobster meat mixed with mayonnaise or butter, served in a toasted hot dog–style bun. A summertime favorite by the seaside.",
            "Macaroni and Cheese": "A beloved comfort food, especially in the U.S. Made by mixing elbow macaroni with a creamy cheese sauce, then baked or served on the stovetop. Often topped with breadcrumbs.",
            "Macarons": "A delicate French confection made of almond meringue cookies filled with flavored buttercream, ganache, or jam. Known for their smooth shells, chewy texture, and pastel colors.",
            "Miso Soup": "A traditional Japanese soup made with dashi (broth) and miso paste, often containing tofu, seaweed, and green onions. Commonly served as a side in Japanese meals.",
            "Mussels": "Shellfish steamed or cooked in sauces like white wine, garlic, and herbs. Popular in European cuisines, especially Belgian moules-frites (mussels with frie",
            "Nachos": "A Tex-Mex dish made from tortilla chips topped with melted cheese and various toppings like jalapeños, beans, sour cream, salsa, guacamole, or meat. A popular snack or party food.",
            "Omelette": "Beaten eggs cooked in a pan, sometimes folded around fillings like cheese, vegetables, or ham. A versatile and popular breakfast dish worldwide.",
            "Onion Rings": "Sliced onions dipped in batter or breadcrumbs and deep-fried until crispy. A common side dish in American fast food.",
            "Oysters": "Shellfish that can be eaten raw on the half-shell (with lemon or mignonette sauce) or cooked in various ways (grilled, fried, baked). Considered a delicacy and sometimes associated with aphrodisiac qualities.",
            "Pad Thai": "A famous Thai stir-fried noodle dish made with rice noodles, eggs, tofu or shrimp, bean sprouts, peanuts, and tamarind-based sauce. Sweet, savory, and tangy.",
            "Paella": "A Spanish rice dish originally from Valencia, made with saffron-flavored rice and a variety of ingredients such as seafood, chicken, rabbit, and vegetables. Cooked in a wide, shallow pan.",
            "Pancakes": "Thin, flat cakes made from a batter of flour, eggs, milk, and butter, cooked on a griddle or frying pan. Served with sweet or savory toppings, popular for breakfast.",
            "Panna Cotta": "An Italian dessert made from sweetened cream thickened with gelatin and molded. Often served with berries, caramel, or chocolate sauce.",
            "Peking Duck": "A famous Chinese dish from Beijing featuring crispy roasted duck served with thin pancakes, hoisin sauce, and sliced scallions.",
            "Pho": "A Vietnamese noodle soup consisting of broth, rice noodles, herbs, and meat, usually beef or chicken. Known for its aromatic and flavorful broth.",
            "Pizza": "A popular Italian dish made with a round, flat base of dough topped with tomato sauce, cheese, and various toppings, then baked. Variations include Neapolitan, New York, and Chicago styles.",
            "Pork Chop": "A cut of pork taken from the loin, typically grilled, pan-fried, or baked. Can be served with a variety of sauces and sides.",
            "Poutine": "A Canadian dish of French fries topped with cheese curds and smothered in gravy. A hearty and indulgent comfort food.",
            "Prime Rib": "A classic roast beef dish, prime rib is a cut from the rib section, known for its tenderness and rich flavor. Often served with au jus and horseradish sauce.",
            "Pulled Pork Sandwich": "Slow-cooked, shredded pork shoulder served on a bun, often with barbecue sauce and coleslaw. A staple of Southern U.S. cuisine.",
            "Ramen": "A Japanese noodle soup with Chinese-style wheat noodles served in a meat or fish-based broth, often flavored with soy sauce or miso, and topped with ingredients like sliced pork, nori, and green onions.",
            "Ravioli": "Italian pasta parcels filled with cheese, meat, or vegetables, served with a sauce such as marinara, cream, or butter and sage.",
            "Red Velvet Cake": "A rich, moist cake with a distinctive red color, typically layered with cream cheese frosting. Popular in the Southern United States.",
            "Risotto": "A creamy Italian rice dish cooked with broth until it reaches a rich, velvety consistency. Often includes ingredients like mushrooms, seafood, or saffron.",
            "Samosa": "A fried or baked pastry with a savory filling, such as spiced potatoes, onions, peas, or lentils. Popular in South Asian cuisine.",
            "Sashimi": "Thinly sliced raw fish or seafood, served without rice. A Japanese delicacy often accompanied by soy sauce, wasabi, and pickled ginger.",
            "Scallops": "Tender, sweet shellfish that can be seared, grilled, or baked. Often served as an appetizer or main course.",
            "Seaweed Salad": "A salad made from various types of seaweed, typically dressed with sesame oil, vinegar, and sesame seeds. Common in Japanese cuisine.",
            "Shrimp and Grits": "A traditional Southern U.S. dish. It features shrimp sautéed with seasonings (often garlic, bacon, butter, or spices) served over creamy grits, which are made from ground corn. Originally a simple fisherman’s breakfast, it’s now considered a comfort food and is often elevated with rich sauces.",
            "Spaghetti Bolognese": "An Italian pasta dish with a rich, slow-cooked meat sauce made from ground beef, tomatoes, onions, and herbs. Typically served over spaghetti noodles and topped with Parmesan cheese.",
            "Spaghetti Carbonara": "An Italian pasta dish made with eggs, cheese (Pecorino Romano or Parmesan), pancetta or guanciale, and black pepper. The ingredients create a creamy sauce without the use of cream.",
            "Spring Rolls": "A popular appetizer in East Asian and Southeast Asian cuisine. They consist of a thin pastry filled with vegetables, meat, or seafood, then rolled and fried or served fresh.",
            "Steak": "A slice of high-quality beef, typically grilled, pan-fried, or broiled. Can be cooked to various levels of doneness and often seasoned or marinated.",
            "Strawberry Shortcake": "A dessert made with layers of sweet biscuits or sponge cake, fresh strawberries, and whipped cream.",
            "Sushi": "A Japanese dish consisting of vinegared rice combined with ingredients like raw fish, seafood, vegetables, or egg. Often served with soy sauce, wasabi, and pickled ginger.",
            "Tacos": "A traditional Mexican dish made with soft or crispy tortillas filled with a variety of ingredients such as seasoned meat, beans, cheese, lettuce, salsa, or guacamole. Street-style tacos are usually small and served with onions, cilantro, and lime.",
            "Takoyaki": "A Japanese street food snack made of a wheat-flour batter cooked in a special round mold, filled with diced octopus, pickled ginger, and green onions. The balls are brushed with takoyaki sauce, drizzled with mayonnaise, and topped with bonito flakes and seaweed powder.",
            "Tiramisu": "A famous Italian dessert made of layers of coffee-soaked ladyfinger biscuits, mascarpone cream, and cocoa powder. Its name means “pick me up,” referring to the energizing coffee flavor. Rich, creamy, and slightly bitter-sweet.",
            "Tuna Tartare": "A raw seafood dish made with finely chopped fresh tuna, often seasoned with soy sauce, sesame oil, citrus, avocado, or herbs. Inspired by beef tartare but lighter and fresher, it’s usually served as an appetizer.",
            "Waffles": "A breakfast or dessert dish made from a batter cooked in a waffle iron, creating a crisp, patterned surface. They can be topped with butter, syrup, fruit, whipped cream, or even fried chicken in savory versions. Belgian waffles are particularly famous for their deep pockets."
            }

            if food_name in descriptions:
                return descriptions[food_name]
            else:
                return "Deskripsi belum tersedia untuk makanan ini."

        desc = get_food_description(predicted_food)
        st.info(f"🔍 {desc}")

        #Buat barplot top 5 prediksi
        top5_labels = [food_labels[i] for i in top5_idx]
        top5_probs = [pred[0][i]*100 for i in top5_idx]

        fig, ax = plt.subplots(figsize=(10, 5))  

        bars = ax.barh(top5_labels, top5_probs, color="skyblue")

        for bar, prob in zip(bars, top5_probs):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    f"{prob:.2f}%", va='center')

        ax.set_xlabel("Probability (%)")
        ax.set_xlim(0, 100)  
        plt.gca().invert_yaxis()  

        st.pyplot(fig)


with tab2:
    st.subheader("About This App")
    st.markdown(
        """
        This app uses a deep learning model to classify images of food into various categories.
        Upload an image, and the model will predict the type of food along with a confidence score.
        """
    )
    st.markdown(
        """
        **Model Details:**
        - Architecture: InceptionV3
        - Dataset: Food-101
        - Number of Classes: 101
        """
    )

    st.write("")

    st.markdown("""**Food Classes:**""")

    st.write("We are using the Food-101 dataset, which includes only the following 101 food categories for image classification.")

    food_list = "\n".join([f"- {food}" for food in food_labels])
    st.markdown(food_list)