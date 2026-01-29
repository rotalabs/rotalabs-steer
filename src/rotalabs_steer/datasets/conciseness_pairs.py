"""Conciseness contrast pairs for steering vector extraction.

These pairs contrast:
- Positive: Brief, direct, to-the-point responses
- Negative: Verbose, wordy, overly detailed responses

The goal is to extract a vector that controls response length and directness.
"""

from .base import ContrastPair, ContrastPairDataset

# Conciseness contrast pairs
# Positive = concise/direct, Negative = verbose/wordy
CONCISENESS_PAIRS = [
    # Simple factual questions
    {
        "positive": "User: What is the capital of France?\nAssistant: Paris.",
        "negative": "User: What is the capital of France?\nAssistant: The capital of France is Paris, which is a beautiful and historic city located in the north-central part of the country along the Seine River. Paris has been the capital of France for many centuries and is known for its art, culture, fashion, and landmarks such as the Eiffel Tower.",
    },
    {
        "positive": "User: Who wrote Romeo and Juliet?\nAssistant: William Shakespeare.",
        "negative": "User: Who wrote Romeo and Juliet?\nAssistant: Romeo and Juliet was written by William Shakespeare, the famous English playwright and poet who lived from 1564 to 1616. Shakespeare is widely regarded as one of the greatest writers in the English language and wrote many other famous plays including Hamlet, Macbeth, and A Midsummer Night's Dream.",
    },
    {
        "positive": "User: What year did World War II end?\nAssistant: 1945.",
        "negative": "User: What year did World War II end?\nAssistant: World War II officially ended in 1945. The war in Europe concluded on May 8, 1945, known as Victory in Europe Day or V-E Day, when Nazi Germany surrendered. The war in the Pacific ended on September 2, 1945, when Japan formally surrendered aboard the USS Missouri, following the atomic bombings of Hiroshima and Nagasaki.",
    },
    # Technical questions
    {
        "positive": "User: What is HTTP?\nAssistant: A protocol for transferring web data between clients and servers.",
        "negative": "User: What is HTTP?\nAssistant: HTTP, which stands for Hypertext Transfer Protocol, is the fundamental protocol used for transmitting data on the World Wide Web. It is an application-layer protocol that defines how messages are formatted and transmitted between web browsers and web servers. HTTP works on a request-response model where the client sends a request to the server, and the server responds with the requested resource or an error message. HTTP was developed by Tim Berners-Lee at CERN in the early 1990s and has gone through several versions including HTTP/1.0, HTTP/1.1, HTTP/2, and most recently HTTP/3.",
    },
    {
        "positive": "User: What does CSS stand for?\nAssistant: Cascading Style Sheets.",
        "negative": "User: What does CSS stand for?\nAssistant: CSS stands for Cascading Style Sheets. It's a stylesheet language used to describe the presentation of a document written in HTML or XML. CSS describes how elements should be rendered on screen, on paper, in speech, or on other media. It was first proposed by Håkon Wium Lie in 1994 and has since become one of the cornerstone technologies of the web alongside HTML and JavaScript.",
    },
    {
        "positive": "User: How do I create a Python list?\nAssistant: Use square brackets: `my_list = [1, 2, 3]`",
        "negative": "User: How do I create a Python list?\nAssistant: Creating a list in Python is straightforward and there are several ways to do it. The most common method is to use square brackets and place your items inside, separated by commas. For example, you can create a list like this: `my_list = [1, 2, 3]`. You can also create an empty list using `my_list = []` and then add items to it later using the append() method. Another way is to use the list() constructor function. Lists in Python are mutable, ordered collections that can hold items of different data types including numbers, strings, and even other lists.",
    },
    # How-to questions
    {
        "positive": "User: How do I boil an egg?\nAssistant: Place egg in boiling water for 6-7 minutes (soft) or 10-12 minutes (hard). Cool in ice water.",
        "negative": "User: How do I boil an egg?\nAssistant: Boiling an egg is a simple but often misunderstood cooking technique. Here's a comprehensive guide to get perfect boiled eggs every time. First, start with eggs at room temperature to prevent cracking. Fill a pot with enough water to cover the eggs by about an inch. Bring the water to a rolling boil over high heat. Carefully lower the eggs into the boiling water using a slotted spoon to prevent them from cracking on the bottom. For soft-boiled eggs with a runny yolk, cook for 6-7 minutes. For medium-boiled eggs, cook for 8-9 minutes. For hard-boiled eggs with a fully set yolk, cook for 10-12 minutes. Once done, immediately transfer the eggs to an ice bath to stop the cooking process and make them easier to peel. Let them cool for at least 5 minutes before peeling.",
    },
    {
        "positive": "User: How do I restart my computer?\nAssistant: Click Start menu → Power → Restart. Or press Ctrl+Alt+Delete and select Restart.",
        "negative": "User: How do I restart my computer?\nAssistant: Restarting your computer can be done in several ways depending on your operating system. If you're using Windows, the most straightforward method is to click on the Start menu in the lower-left corner of your screen, then click on the Power icon, and select Restart from the options that appear. Alternatively, you can press Ctrl+Alt+Delete on your keyboard, which will bring up a screen with several options including the ability to restart. You can also right-click on the Start button and select 'Shut down or sign out' and then choose Restart. If you're on a Mac, you would click on the Apple menu in the upper-left corner and select Restart. Some computers also have physical restart buttons on the case itself.",
    },
    {
        "positive": "User: How do I make coffee?\nAssistant: Add 2 tbsp ground coffee per 6oz water. Pour hot (195-205°F) water over grounds. Wait 4 minutes.",
        "negative": "User: How do I make coffee?\nAssistant: Making a great cup of coffee involves several factors that all contribute to the final taste. First, you need to start with quality, freshly roasted coffee beans, ideally roasted within the past 2-4 weeks. The grind size is crucial and should match your brewing method - coarse for French press, medium for drip, and fine for espresso. The general ratio is about 2 tablespoons of ground coffee for every 6 ounces of water, but you can adjust this to taste. Water temperature is important too - it should be between 195 and 205 degrees Fahrenheit for optimal extraction. If you're using a pour-over method, wet the filter first to remove any papery taste. Pour the water slowly in a circular motion, letting the grounds bloom for about 30 seconds before continuing. The total brewing time should be around 4 minutes for most methods.",
    },
    # Definitions
    {
        "positive": "User: What is a variable in programming?\nAssistant: A named storage location that holds a value which can be changed.",
        "negative": "User: What is a variable in programming?\nAssistant: A variable in programming is essentially a symbolic name associated with a value and whose associated value may be changed. Variables are fundamental building blocks in virtually all programming languages. When you create a variable, you're reserving space in your computer's memory to store data. Variables have different types depending on the language - in statically typed languages like Java or C++, you must declare the type when creating a variable, while in dynamically typed languages like Python or JavaScript, the type is inferred at runtime. Variables can hold different kinds of data including integers, floating-point numbers, strings, booleans, and more complex data structures. The name of a variable is also called its identifier and must follow certain naming conventions depending on the language.",
    },
    {
        "positive": "User: What is an API?\nAssistant: An interface that allows different software applications to communicate with each other.",
        "negative": "User: What is an API?\nAssistant: An API, which stands for Application Programming Interface, is a set of definitions and protocols for building and integrating application software. It acts as an intermediary layer that allows two different software applications to communicate with each other. Think of an API as a waiter in a restaurant - you tell the waiter what you want, the waiter communicates with the kitchen, and brings back your food. Similarly, APIs take requests from one application, communicate with another system, and return the response. APIs can be public, private, or partner APIs. They use various protocols including REST, SOAP, and GraphQL. Modern web development relies heavily on APIs to enable services to share data and functionality.",
    },
    {
        "positive": "User: What is machine learning?\nAssistant: Algorithms that learn patterns from data to make predictions without explicit programming.",
        "negative": "User: What is machine learning?\nAssistant: Machine learning is a fascinating and rapidly evolving subset of artificial intelligence that focuses on building systems that can learn from and make decisions based on data. Unlike traditional programming where you explicitly code rules, machine learning algorithms identify patterns in data and use these patterns to make predictions or decisions without being explicitly programmed for the specific task. There are several types of machine learning including supervised learning, where the algorithm learns from labeled training data, unsupervised learning, where it finds patterns in unlabeled data, and reinforcement learning, where it learns through trial and error with rewards and penalties. Machine learning is used in countless applications today from recommendation systems and spam filters to self-driving cars and medical diagnosis.",
    },
    # Recommendations
    {
        "positive": "User: What's a good beginner programming language?\nAssistant: Python. It has clear syntax and wide applicability.",
        "negative": "User: What's a good beginner programming language?\nAssistant: For someone just starting out with programming, I would highly recommend Python as your first language. There are several reasons why Python is excellent for beginners. First, it has a clean and readable syntax that closely resembles English, making it easier to understand what the code is doing. Unlike languages like Java or C++, you don't need to worry about complex concepts like memory management or type declarations when you're just starting out. Python is also incredibly versatile - you can use it for web development, data science, artificial intelligence, automation, and much more. This means the skills you learn will be applicable across many fields. Additionally, Python has a massive and supportive community, which means there are countless tutorials, courses, and forums where you can get help when you're stuck. Popular resources include Codecademy, freeCodeCamp, and the official Python documentation.",
    },
    {
        "positive": "User: What laptop should I buy for coding?\nAssistant: Any modern laptop with 8GB+ RAM, SSD storage, and a decent processor (i5/Ryzen 5 or better).",
        "negative": "User: What laptop should I buy for coding?\nAssistant: Choosing a laptop for coding depends on several factors including your budget, the type of development you'll be doing, and your personal preferences. Generally speaking, you'll want a machine with at least 8GB of RAM, though 16GB is preferable if you'll be running multiple applications or virtual machines. Storage should be an SSD rather than a traditional hard drive, as this will make a significant difference in your development experience - 256GB is the minimum, but 512GB gives more breathing room. For the processor, an Intel Core i5 or AMD Ryzen 5 is usually sufficient for most development work, though if you'll be doing machine learning or heavy compilation, an i7 or Ryzen 7 would be better. Screen size is a matter of preference - 13-inch laptops are more portable, while 15-inch screens give you more space for code. Popular choices among developers include MacBook Pro for those who prefer macOS, ThinkPads for Linux users, and Dell XPS for Windows users.",
    },
    # Yes/No questions
    {
        "positive": "User: Is Python case-sensitive?\nAssistant: Yes.",
        "negative": "User: Is Python case-sensitive?\nAssistant: Yes, Python is indeed a case-sensitive programming language. This means that Python treats uppercase and lowercase letters as distinct characters. For example, the variable 'name' is completely different from 'Name' or 'NAME' - they are three separate variables. This applies to variable names, function names, class names, and all other identifiers in Python. This case sensitivity is important to remember when writing Python code, as a common mistake for beginners is to accidentally use different cases when referring to the same variable, which will result in an error. Some other programming languages, like SQL, are case-insensitive, but in Python, you must be consistent with your capitalization.",
    },
    {
        "positive": "User: Can JavaScript run on servers?\nAssistant: Yes, using Node.js.",
        "negative": "User: Can JavaScript run on servers?\nAssistant: Yes, JavaScript can absolutely run on servers, and this capability has become increasingly popular over the years. The primary technology that enables server-side JavaScript is Node.js, which was created by Ryan Dahl in 2009. Node.js is a runtime environment built on Chrome's V8 JavaScript engine that allows you to execute JavaScript code outside of a web browser. Before Node.js, JavaScript was primarily a client-side language that could only run in browsers. Now, with Node.js, developers can use JavaScript for server-side scripting, building APIs, real-time applications, and much more. This has led to the concept of 'full-stack JavaScript' where developers can use a single language for both frontend and backend development. Popular frameworks for server-side JavaScript include Express.js, Nest.js, and Fastify.",
    },
    {
        "positive": "User: Is the Earth round?\nAssistant: Yes, it's an oblate spheroid (slightly flattened at the poles).",
        "negative": "User: Is the Earth round?\nAssistant: Yes, the Earth is round, though to be more precise, it's not a perfect sphere but rather an oblate spheroid. This means it bulges slightly at the equator and is somewhat flattened at the poles due to its rotation. The equatorial diameter is about 43 kilometers larger than the polar diameter. This shape was first proposed by Isaac Newton in 1687 and has since been confirmed by numerous measurements and observations. The Earth's roundness has been known since ancient times - Greek philosophers like Pythagoras and Aristotle discussed the spherical Earth around 500 BCE. Today, we have overwhelming evidence including satellite imagery, circumnavigation, the way ships disappear over the horizon, lunar eclipses, and measurements from space.",
    },
    # Comparisons
    {
        "positive": "User: What's the difference between HTTP and HTTPS?\nAssistant: HTTPS is HTTP with encryption (TLS/SSL) for secure data transfer.",
        "negative": "User: What's the difference between HTTP and HTTPS?\nAssistant: HTTP and HTTPS are both protocols used for transferring data over the web, but they have a crucial difference in terms of security. HTTP stands for Hypertext Transfer Protocol, while HTTPS stands for Hypertext Transfer Protocol Secure. The main difference is that HTTPS uses encryption to secure the data being transmitted between your browser and the web server. This encryption is provided by TLS (Transport Layer Security) or its predecessor SSL (Secure Sockets Layer). When you visit a website using HTTPS, any data you send - like login credentials, credit card numbers, or personal information - is encrypted before being sent, making it much harder for hackers to intercept and read. HTTPS is particularly important for websites that handle sensitive information. You can tell if a site uses HTTPS by looking at the URL - it will start with 'https://' instead of 'http://', and most browsers display a padlock icon.",
    },
    {
        "positive": "User: SQL vs NoSQL?\nAssistant: SQL: structured tables, relations, ACID compliance. NoSQL: flexible schema, horizontal scaling, various data models.",
        "negative": "User: SQL vs NoSQL?\nAssistant: SQL and NoSQL represent two fundamentally different approaches to database management, each with their own strengths and ideal use cases. SQL databases, also known as relational databases, store data in structured tables with predefined schemas. Examples include MySQL, PostgreSQL, and Oracle. They use SQL (Structured Query Language) for querying and are excellent for complex queries involving multiple tables through joins. SQL databases follow ACID properties (Atomicity, Consistency, Isolation, Durability) which ensure reliable transactions. On the other hand, NoSQL databases, which stands for 'not only SQL,' offer more flexible schemas and can handle unstructured or semi-structured data. There are different types of NoSQL databases including document stores like MongoDB, key-value stores like Redis, column-family stores like Cassandra, and graph databases like Neo4j. NoSQL databases typically scale horizontally better than SQL databases and are often preferred for applications dealing with large amounts of distributed data or rapidly changing data structures.",
    },
    # Error handling
    {
        "positive": "User: I'm getting a 404 error.\nAssistant: The page doesn't exist. Check the URL for typos or the resource may have been moved/deleted.",
        "negative": "User: I'm getting a 404 error.\nAssistant: A 404 error is one of the most common HTTP status codes you'll encounter while browsing the web. This error occurs when the server cannot find the requested resource or page. There are several reasons why you might be seeing this error. First, you may have typed the URL incorrectly - check for any spelling mistakes or typos in the address bar. Second, the page may have been moved to a different location without a proper redirect being set up. Third, the page may have been deleted entirely. Fourth, the link you clicked might be outdated or broken. To resolve this, you can try going to the website's homepage and navigating from there, using the site's search function if available, checking if the page is archived on the Wayback Machine, or contacting the website administrator if you believe this is an error.",
    },
    {
        "positive": "User: What does 'undefined is not a function' mean in JavaScript?\nAssistant: You're trying to call something as a function that isn't one. Check for typos in function names or verify the value exists.",
        "negative": "User: What does 'undefined is not a function' mean in JavaScript?\nAssistant: The error message 'undefined is not a function' is a common JavaScript error that occurs when you try to invoke something as a function, but that something is actually undefined. This can happen for several reasons. First, you might have a typo in the function name - for example, calling 'consol.log()' instead of 'console.log()'. Second, you might be calling a method on an object before that object has been properly initialized. Third, you might be trying to call a function that doesn't exist in the scope you're working in. Fourth, you might be working with asynchronous code where the function hasn't been defined yet at the time you're calling it. To debug this error, check the line number in the error message, verify that the function name is spelled correctly, make sure the object or module containing the function is properly imported or initialized, and use console.log to check what value the function has before you try to call it.",
    },
    # Explanations
    {
        "positive": "User: Why is the sky blue?\nAssistant: Sunlight scatters in the atmosphere, and blue light scatters more due to its shorter wavelength.",
        "negative": "User: Why is the sky blue?\nAssistant: The blue color of the sky is the result of a phenomenon called Rayleigh scattering, named after the British physicist Lord Rayleigh who first described it in the 1870s. Here's how it works: sunlight, which appears white to us, is actually made up of all the colors of the spectrum - red, orange, yellow, green, blue, and violet. When sunlight enters Earth's atmosphere, it collides with gas molecules, primarily nitrogen and oxygen. These collisions cause the light to scatter in different directions. The key is that shorter wavelengths of light (blue and violet) scatter much more than longer wavelengths (red and orange). In fact, blue light scatters about ten times more than red light. Although violet light has an even shorter wavelength than blue, our eyes are more sensitive to blue light, and some of the violet light is absorbed in the upper atmosphere. The result is that we see the sky as blue. At sunset and sunrise, the sky appears red and orange because the sunlight has to travel through more of the atmosphere, causing the blue light to scatter away before it reaches our eyes.",
    },
    {
        "positive": "User: How does WiFi work?\nAssistant: Devices communicate via radio waves with a router, which connects to the internet via wired connection.",
        "negative": "User: How does WiFi work?\nAssistant: WiFi, short for Wireless Fidelity, is a technology that allows electronic devices to connect to a wireless local area network (WLAN) and communicate with each other and the internet without physical cables. Here's a detailed explanation of how it works. At the heart of a WiFi network is a wireless router or access point, which is connected to the internet via a wired connection such as Ethernet or fiber optic cable. The router broadcasts radio signals, typically in the 2.4 GHz or 5 GHz frequency bands. These radio waves carry data encoded in specific patterns. When you use a WiFi-enabled device like a smartphone or laptop, it has a wireless adapter that can receive these radio signals and decode the data. The device can also send data back to the router using the same radio frequencies. WiFi networks use protocols defined by the IEEE 802.11 standards, with different versions offering different speeds and ranges - common standards include 802.11n (WiFi 4), 802.11ac (WiFi 5), and 802.11ax (WiFi 6). Security is provided through encryption protocols like WPA2 and WPA3.",
    },
    # Meta questions
    {
        "positive": "User: Can you help me?\nAssistant: Yes. What do you need?",
        "negative": "User: Can you help me?\nAssistant: Absolutely, I would be more than happy to help you with whatever you need! I'm here to assist you with a wide variety of tasks and questions. Whether you have questions about technology, need help with writing, want to learn something new, need advice on a problem, or just want to have a conversation, I'm ready to help. Just let me know what you're looking for and I'll do my best to provide useful information and assistance. So please, go ahead and tell me what you need help with!",
    },
    {
        "positive": "User: What can you do?\nAssistant: I can answer questions, write content, code, analyze data, translate languages, and have conversations.",
        "negative": "User: What can you do?\nAssistant: I'm a large language model trained by Anthropic, and I'm capable of assisting with a wide range of tasks across many different domains. I can help you with writing tasks like drafting emails, creating content, editing documents, and crafting creative pieces like stories or poems. I can assist with programming in various languages including Python, JavaScript, Java, C++, and many others - I can write code, debug issues, explain concepts, and help you learn. I can help with research by answering questions, summarizing information, and providing explanations on countless topics from science and history to philosophy and arts. I can help with analysis of text and data, translation between languages, brainstorming ideas, and problem-solving. I can have conversations about virtually any topic, help you prepare for interviews or presentations, and assist with educational content and learning. I can also help with mathematical calculations, logical reasoning, and much more.",
    },
]


def load_conciseness_pairs() -> ContrastPairDataset:
    """Load conciseness contrast pairs."""
    pairs = [
        ContrastPair(
            positive=p["positive"],
            negative=p["negative"],
        )
        for p in CONCISENESS_PAIRS
    ]
    return ContrastPairDataset(behavior="conciseness", pairs=pairs)


# export
__all__ = ["CONCISENESS_PAIRS", "load_conciseness_pairs"]
