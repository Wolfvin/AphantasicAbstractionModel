"""
RSVS v0.9 — Embedded corpus for demonstration and benchmarking.

Wikipedia-style texts covering multiple domains.
Used when live Wikipedia access is unavailable.
Each domain is a list of sentences extracted/paraphrased from Wikipedia articles.
"""

DOMAINS = {

    "geology": [
        "Rock is a naturally occurring solid aggregate of one or more minerals.",
        "Stone is a hard solid mineral material found in the earth.",
        "Granite is a coarse-grained igneous rock composed mainly of quartz and feldspar.",
        "Basalt is a dark fine-grained volcanic rock that is the most common type of solid rock.",
        "Sedimentary rock is formed by the deposition and cementation of mineral particles.",
        "Limestone is a sedimentary rock composed mainly of calcium carbonate.",
        "Marble is a metamorphic rock composed of recrystallized carbonate minerals.",
        "Quartz is the second most abundant mineral in the continental crust.",
        "Feldspar is the most abundant mineral group in the earth's crust.",
        "Erosion is the process by which rock and soil are worn away by wind and water.",
        "Weathering breaks down rock through physical and chemical processes over time.",
        "A mineral is a naturally occurring inorganic solid with a crystalline structure.",
        "Crystals form when a mineral solidifies slowly from a liquid state.",
        "Volcanic rock is formed when magma erupts from a volcano and cools rapidly.",
        "Igneous rock forms through the cooling and solidification of magma or lava.",
        "Metamorphic rock forms when existing rock is changed by heat and pressure.",
        "The rock cycle describes how rocks change from one type to another over time.",
        "Sandstone is a sedimentary rock composed of sand-sized mineral particles.",
        "Shale is a fine-grained sedimentary rock formed from clay and silt particles.",
        "Obsidian is a naturally occurring volcanic glass formed from rapidly cooling lava.",
        "Coal is a combustible black sedimentary rock formed from plant material.",
        "Diamond is the hardest natural material and a form of carbon crystal.",
        "Gold is a dense soft shiny metallic element that is highly resistant to corrosion.",
        "Iron is the most common element on Earth by mass and a hard metallic substance.",
        "Copper is a soft malleable ductile metal with very high thermal and electrical conductivity.",
        "A fossil is the preserved remains of a once-living organism found in rock.",
        "Tectonic plates are large segments of the earth's crust that move slowly over time.",
        "An earthquake occurs when tectonic plates suddenly shift releasing stored energy.",
        "A volcano is an opening in the earth's crust through which lava and gases escape.",
        "Soil is a mixture of organic matter minerals gases liquids and organisms.",
    ],

    "water": [
        "Water is a transparent tasteless odorless nearly colorless chemical substance.",
        "Water is composed of hydrogen and oxygen atoms bonded together.",
        "Ice is the solid form of water and forms when water freezes below zero degrees.",
        "Steam is the gaseous form of water produced when water is heated to boiling.",
        "A river is a large natural flowing watercourse moving toward an ocean or lake.",
        "An ocean is a vast body of salt water covering most of the earth's surface.",
        "Rain is liquid water falling from clouds in the atmosphere.",
        "Snow is precipitation in the form of ice crystals falling from clouds.",
        "A lake is a large body of water surrounded by land.",
        "A waterfall is a point where water flows over a vertical drop in a river.",
        "Evaporation is the process by which liquid water transforms into water vapor.",
        "Condensation is the process by which water vapor transforms into liquid water.",
        "The water cycle describes the continuous movement of water on Earth.",
        "Groundwater is water found underground in the cracks and spaces in soil and rock.",
        "A glacier is a large persistent body of dense ice that moves slowly over land.",
        "Water is essential for all known forms of life to survive and function.",
        "Freshwater is water with low concentrations of dissolved salts.",
        "Saltwater contains dissolved salts and minerals making it unfit for drinking.",
        "Waves are the movement of energy through water caused by wind or geological activity.",
        "Tides are the regular rise and fall of sea levels caused by the moon's gravity.",
        "A flood occurs when water overflows its normal boundaries and inundates land.",
        "Drought is a prolonged period of abnormally low rainfall.",
        "Humidity is the amount of water vapor present in the air.",
        "Aquatic means living or growing in or near water.",
        "A wetland is a land area that is saturated with water either permanently or seasonally.",
        "Coral reefs are underwater structures made of calcium carbonate secreted by corals.",
        "Pressure increases with depth in water due to the weight of the water above.",
        "Surface tension is the tendency of liquid surfaces to shrink to the minimum area.",
        "Water is an excellent solvent and dissolves many substances readily.",
        "The density of water is approximately one gram per cubic centimeter.",
    ],

    "biology": [
        "A cell is the basic structural and functional unit of all known living organisms.",
        "DNA is a molecule that carries the genetic instructions for living organisms.",
        "Proteins are large complex molecules that carry out most of the work in cells.",
        "Photosynthesis is the process by which plants convert light energy into chemical energy.",
        "Respiration is the process by which organisms convert oxygen and glucose into energy.",
        "Evolution is the change in heritable traits of populations over successive generations.",
        "Natural selection is the mechanism by which organisms with favorable traits reproduce more.",
        "A species is a group of organisms capable of interbreeding and producing fertile offspring.",
        "An ecosystem is a community of living organisms interacting with their environment.",
        "A food chain shows how energy flows from one organism to another through feeding.",
        "A predator is an organism that hunts and feeds on other organisms called prey.",
        "Bacteria are single-celled microorganisms that are among the earliest life forms on Earth.",
        "A virus is an infectious agent that replicates inside living cells of organisms.",
        "Fungi are a kingdom of organisms including molds yeasts and mushrooms.",
        "Plants are multicellular organisms that produce energy through photosynthesis.",
        "Animals are multicellular organisms that consume food for energy and movement.",
        "Mammals are warm-blooded vertebrates that nurse their young with milk.",
        "Birds are warm-blooded egg-laying vertebrates with feathers and beaks.",
        "Fish are cold-blooded aquatic vertebrates with gills and fins.",
        "Insects are the largest class of animals with three body segments and six legs.",
        "A habitat is the natural environment in which an organism lives.",
        "Biodiversity refers to the variety of life found on Earth.",
        "Extinction is the termination of a species when the last individual dies.",
        "A gene is a sequence of DNA that contains instructions for making a protein.",
        "Heredity is the passing of traits from parents to offspring through genes.",
        "A hormone is a chemical messenger produced by glands that regulates body functions.",
        "The nervous system controls and coordinates body functions through electrical signals.",
        "The immune system protects the body against pathogens and disease.",
        "Metabolism refers to all chemical reactions that occur in a living organism.",
        "A symbiosis is a close long-term interaction between two different biological species.",
    ],

    "physics": [
        "Force is an interaction that changes the motion of an object.",
        "Mass is the amount of matter contained in an object.",
        "Energy is the capacity to do work and exists in many forms.",
        "Heat is a form of energy that flows from a hot object to a cold one.",
        "Light is electromagnetic radiation that is visible to the human eye.",
        "Sound is a mechanical wave that propagates through a medium such as air.",
        "Gravity is the force that attracts objects with mass toward one another.",
        "Electricity is the flow of electric charge through a conductor.",
        "Magnetism is a force exerted by magnets and electric currents.",
        "Temperature is a measure of the average kinetic energy of particles.",
        "Pressure is force applied per unit area on a surface.",
        "Velocity is the rate of change of position with respect to time.",
        "Acceleration is the rate of change of velocity with respect to time.",
        "A wave is a disturbance that transfers energy through matter or space.",
        "Friction is a force that opposes relative motion between surfaces in contact.",
        "Density is mass per unit volume of a substance.",
        "Buoyancy is the upward force exerted on an object submerged in a fluid.",
        "Thermal expansion is the tendency of matter to change volume in response to temperature.",
        "Radiation is energy emitted as electromagnetic waves or subatomic particles.",
        "An atom is the smallest unit of an element that retains its chemical properties.",
        "A molecule is two or more atoms bonded together as the smallest unit of a substance.",
        "Chemical bonding is the attraction between atoms that allows the formation of molecules.",
        "A solid has a definite shape and volume with particles closely packed together.",
        "A liquid has a definite volume but no definite shape taking the shape of its container.",
        "A gas has no definite shape or volume and expands to fill its container.",
        "Plasma is a high-temperature state of matter consisting of ionized gas.",
        "Nuclear energy is released through fission or fusion of atomic nuclei.",
        "Inertia is the resistance of an object to changes in its state of motion.",
        "Conservation of energy states that energy cannot be created or destroyed only converted.",
        "Quantum mechanics describes the behavior of matter and energy at atomic scales.",
    ],

    "materials": [
        "A material is a substance from which objects can be made.",
        "Metal is a solid material that is typically hard shiny malleable and ductile.",
        "Steel is an alloy of iron and carbon that is strong and widely used in construction.",
        "Aluminum is a lightweight silvery metal with high strength and corrosion resistance.",
        "Concrete is a composite material made of aggregate bonded with cement and water.",
        "Wood is a hard fibrous structural tissue found in trees and other plants.",
        "Plastic is a synthetic material made from polymers that can be molded into any shape.",
        "Glass is an amorphous solid made from silica with a smooth hard transparent surface.",
        "Rubber is an elastic material that returns to its original shape after stretching.",
        "Ceramic is an inorganic non-metallic solid made by heating and cooling.",
        "A composite material is made from two or more constituent materials.",
        "Hardness is the resistance of a material to surface scratching or indentation.",
        "Tensile strength is the maximum stress a material can withstand while being stretched.",
        "Conductivity refers to the ability of a material to conduct heat or electricity.",
        "Insulation is a material that reduces the transfer of heat or electricity.",
        "Brittleness is the tendency of a material to fracture under stress.",
        "Elasticity is the ability of a material to return to its original shape after deformation.",
        "Porosity is the measure of void spaces in a material.",
        "Density is a key physical property of materials affecting strength and weight.",
        "Corrosion is the gradual deterioration of a material through chemical reaction.",
        "Alloy is a mixture of a metal with another element to improve its properties.",
        "Polymer is a large molecule made of many repeating smaller molecules called monomers.",
        "Fiber is a slender threadlike structure that can be woven into textiles.",
        "Adhesive is a substance that bonds surfaces together through surface attachment.",
        "Solubility is the ability of a substance to dissolve in a solvent.",
        "Transparency is the property of allowing light to pass through without scattering.",
        "Surface area affects how quickly a material reacts with its environment.",
        "Thermal conductivity measures how well a material transfers heat.",
        "Malleability is the ability of a material to be shaped by hammering without breaking.",
        "Melting point is the temperature at which a solid becomes liquid.",
    ],

    # --- Domain: profession ---
    # Anchor words: doctor, patient, farmer, teacher, engineer
    "profession": [
        "The doctor treats the patient every day.",
        "The doctor examines the patient carefully.",
        "The farmer grows food for the patient.",
        "The teacher helps the doctor understand science.",
        "The engineer builds tools for the doctor.",
        "The patient trusts the doctor completely.",
        "The farmer and the patient discuss nutrition.",
        "The teacher and the farmer share knowledge.",
        "The engineer and the farmer design equipment.",
        "A doctor prescribes medicine for the patient.",
        "A patient visits the doctor every week.",
        "The farmer feeds the patient fresh produce.",
        "The teacher trains the farmer new methods.",
        "The engineer advises the farmer on irrigation.",
        "The doctor diagnoses the patient accurately.",
        "The patient and the teacher thank the doctor for help.",
        "The farmer supplies food to the patient.",
        "The farmer learns from the teacher regularly.",
        "The farmer hires the engineer for planning.",
        "The engineer supports the doctor with technology.",
        "A doctor saves a patient from illness.",
        "The teacher teaches the patient new skills.",
        "The farmer works harder than the patient.",
        "The teacher guides the engineer through research.",
        "The engineer builds devices for the patient.",
        "The doctor respects the teacher and the farmer.",
        "The patient admires the doctor and the engineer.",
        "The teacher recommends the doctor to the farmer.",
        "The farmer consults the engineer and the doctor.",
        "The engineer collaborates with the teacher and the patient.",
    ],

    # --- Domain: history ---
    # Anchor words: war, empire, trade, civilization, ruler
    "history": [
        "The ruler starts a war against the enemy.",
        "The empire expands after every war.",
        "Trade connects the empire to distant lands.",
        "The civilization flourishes through trade and peace.",
        "A ruler builds the empire with strength.",
        "War destroys many cities in the empire.",
        "Trade enriches the ruler and the civilization.",
        "The ruler controls trade across the empire.",
        "The civilization survives the war and rebuilds.",
        "An empire falls when the ruler loses war.",
        "Trade declines during every major war.",
        "The ruler protects the civilization from invasion.",
        "War weakens the empire and the ruler.",
        "The civilization invents writing during peace before war.",
        "The ruler promotes trade to fund the empire.",
        "A war changes the ruler and the civilization.",
        "The empire monopolizes trade along the river.",
        "The civilization trades goods with the empire.",
        "The ruler declares war on the neighbor.",
        "Trade brings wealth to the civilization and the ruler.",
        "The empire wages war across the continent.",
        "The ruler establishes trade routes for the empire.",
        "War disrupts trade between the empire and rivals.",
        "The civilization records every war and every ruler.",
        "The empire rewards the ruler who wins war.",
        "The ruler unites the civilization after the war.",
        "Trade supports the empire during peace and war.",
        "The civilization honors the ruler and the trade.",
        "The empire remembers every ruler and every war.",
        "The ruler strengthens the civilization through trade and war.",
    ],

    # --- Domain: technology ---
    # Anchor words: computer, network, data, software, processor
    "technology": [
        "The computer processes data through the processor.",
        "The network connects every computer to the server.",
        "The software runs on the computer efficiently.",
        "The processor executes instructions from the software.",
        "A computer stores data for the user.",
        "The network transmits data between computers.",
        "The software manages the network and the data.",
        "A processor powers the computer and the network.",
        "The computer sends data across the network.",
        "The software updates the processor and the computer.",
        "A network routes data to the correct computer.",
        "The processor handles data from the network.",
        "The computer requires software to process data.",
        "The network delivers data for the software.",
        "The processor executes software on the computer.",
        "Data flows from the computer through the network.",
        "The software protects data on the computer.",
        "The processor accelerates the network and the software.",
        "Every computer needs a processor and software.",
        "The network stores data for the software system.",
        "A computer analyzes data using the processor.",
        "The software controls the network and the processor.",
        "Data moves between the computer and the network.",
        "The processor calculates data for the software.",
        "The computer connects the software to the network.",
        "The network relies on the processor for speed.",
        "The software organizes data on the computer disk.",
        "The processor and the computer process data together.",
        "The software compresses data for the network.",
        "The computer and the processor handle the software.",
    ],

    # --- Domain: society ---
    # Anchor words: law, government, citizen, economy, institution
    "society": [
        "The government enforces law for the citizen.",
        "The citizen obeys the law of the government.",
        "The institution supports the government and the economy.",
        "Law protects the citizen and the economy.",
        "The government regulates the economy through law.",
        "The institution teaches the citizen about law.",
        "The economy depends on the government and the institution.",
        "A citizen respects the law and the institution.",
        "The government builds every institution for the citizen.",
        "Law shapes the economy and the government.",
        "The institution enforces law for the government.",
        "The citizen contributes to the economy and the government.",
        "The government passes law to help the citizen.",
        "The economy serves the citizen through the institution.",
        "The institution advises the government on law.",
        "The citizen trusts the government and the law.",
        "The economy grows when the government reforms law.",
        "The institution trains the citizen for the economy.",
        "The government and the institution uphold the law.",
        "The citizen participates in the government through the institution.",
        "Law governs the economy and the institution.",
        "The institution protects the citizen under the law.",
        "The government stabilizes the economy with new law.",
        "Every citizen relies on the institution and the law.",
        "The economy benefits the citizen and the government.",
        "The institution drafts law for the government.",
        "The government funds the institution and the economy.",
        "The citizen follows the law to support the economy.",
        "The institution and the government manage the economy.",
        "The citizen petitions the government to change the law.",
    ],
}

# Flat list of all sentences with domain labels
ALL_SENTENCES = [
    (domain, sentence)
    for domain, sentences in DOMAINS.items()
    for sentence in sentences
]

def get_domain_text(domain: str) -> str:
    """Get all sentences for a domain as a single text block."""
    return " ".join(DOMAINS.get(domain, []))

def get_all_text() -> str:
    """Get all sentences across all domains."""
    return " ".join(s for _, s in ALL_SENTENCES)

def domain_names():
    return list(DOMAINS.keys())


# ---------------------------------------------------------------------------
# Multi-language corpus — English + Indonesian aligned sentences
# for cross-language convergence testing
# ---------------------------------------------------------------------------

# 9 domains covering distinct conceptual areas with natural translations.
# These domains are designed for convergence detection: the same concepts
# expressed in two languages should converge to the same RSVS structure.

CORPUS_EN: dict[str, list[str]] = {
    # --- Domain: royalty (kings, queens, kingdoms) ---
    "royalty": [
        "The king rules over the kingdom with wisdom and authority.",
        "The queen advises the king on matters of state and diplomacy.",
        "A kingdom thrives when the ruler governs with justice and fairness.",
        "The throne passes from the king to the eldest prince by tradition.",
        "The royal court gathers nobles who serve the king and the queen.",
        "A crown symbolizes the power and responsibility of the monarch.",
        "The king commands the army to defend the kingdom from invaders.",
        "The queen oversees the education of the young princesses and princes.",
        "Royal decrees carry the weight of law throughout the kingdom.",
        "The kingdom celebrates when the king returns victorious from battle.",
    ],

    # --- Domain: philosophy (existence, meaning, truth) ---
    "philosophy": [
        "Truth is the ultimate goal of philosophical inquiry and reflection.",
        "Existence precedes essence according to existentialist philosophy.",
        "The meaning of life has been debated by thinkers for millennia.",
        "Reason and logic form the foundation of philosophical argument.",
        "Wisdom emerges from questioning the nature of reality and knowledge.",
        "Ethics examines what constitutes a good and virtuous life.",
        "Consciousness remains one of the deepest mysteries in philosophy.",
        "The search for truth requires doubt and rigorous examination.",
        "Free will and determinism present a fundamental philosophical tension.",
        "A philosopher seeks understanding beyond mere opinion and belief.",
    ],

    # --- Domain: medicine (disease, treatment, healing) ---
    "medicine": [
        "The physician diagnoses the disease through careful examination and testing.",
        "Treatment must address the root cause rather than merely suppressing symptoms.",
        "Healing requires both medical intervention and the body's natural recovery.",
        "Prevention of disease is more effective than any cure after onset.",
        "The patient follows the prescribed regimen to recover from illness.",
        "Surgery is sometimes necessary when medicine alone cannot heal the wound.",
        "Antibiotics fight bacterial infection but cannot treat viral disease.",
        "The hospital provides specialized care for severe and critical conditions.",
        "Research into new treatments advances the frontier of modern medicine.",
        "A diagnosis confirms the nature of the disease and guides therapy.",
    ],

    # --- Domain: nature (rivers, mountains, forests) ---
    "nature": [
        "The river carves a path through the mountain over thousands of years.",
        "Ancient forests shelter countless species within their dense canopy.",
        "The mountain stands as a silent witness to the passage of ages.",
        "A forest purifies the air and regulates the flow of the river.",
        "The valley lies between two mountain ranges fed by a winding river.",
        "Nature balances growth and decay in an endless cycle of renewal.",
        "The river nourishes the forest and the forest protects the river.",
        "Wildlife depends on the forest for shelter and the river for water.",
        "The mountain ecosystem supports life from base to snow-covered peak.",
        "Seasons transform the forest and change the rhythm of the river.",
    ],

    # --- Domain: warfare (strategy, battle, defense) ---
    "warfare": [
        "Strategy determines the outcome of battle before the first shot is fired.",
        "Defense requires fortified positions and well-trained disciplined soldiers.",
        "The general devises a strategy to outmaneuver the enemy in battle.",
        "A siege tests the endurance of both the attacker and the defense.",
        "Victory in battle depends on preparation and superior strategy.",
        "The army strengthens its defense along the vulnerable frontier.",
        "Intelligence and deception are essential elements of warfare strategy.",
        "A battle can turn when one side gains a decisive strategic advantage.",
        "The defense holds the line while the reserves prepare a counterattack.",
        "Throughout history warfare has driven innovation in strategy and technology.",
    ],

    # --- Domain: commerce (trade, market, exchange) ---
    "commerce": [
        "Trade connects distant regions through the exchange of goods and services.",
        "The market sets the price through the balance of supply and demand.",
        "Merchants facilitate commerce by bridging producers and consumers.",
        "An exchange of value is the fundamental transaction in any market.",
        "Commerce flourishes when trade routes are safe and regulations are fair.",
        "The market rewards efficiency and punishes waste in competitive trade.",
        "International trade expands the market beyond domestic borders.",
        "A fair exchange requires transparency and trust between trading partners.",
        "Commerce drives prosperity by enabling specialization and innovation.",
        "The market adjusts prices to reflect changes in supply and demand.",
    ],

    # --- Domain: law (justice, regulation, contract) ---
    "law": [
        "Justice is the foundation upon which the rule of law stands.",
        "A contract binds the parties to their agreed obligations and rights.",
        "Regulation ensures that commerce operates within fair and lawful bounds.",
        "The court interprets the law and delivers justice to the parties.",
        "A legal contract must be voluntary and supported by consideration.",
        "Regulation protects the public from harm and unjust business practices.",
        "The judge applies the law impartially to achieve justice in each case.",
        "A breach of contract entitles the injured party to a legal remedy.",
        "Law evolves through legislation and judicial interpretation over time.",
        "Justice requires that the law treat all persons with equal dignity.",
    ],

    # --- Domain: science (experiment, theory, discovery) ---
    "science": [
        "An experiment tests a hypothesis under controlled and repeatable conditions.",
        "A theory explains observations and predicts the outcome of future experiments.",
        "Discovery advances scientific knowledge by revealing previously unknown phenomena.",
        "The scientific method requires that every theory be falsifiable by experiment.",
        "A well-designed experiment minimizes bias and isolates the variable of interest.",
        "A robust theory withstands repeated experimental testing and peer scrutiny.",
        "Discovery often emerges at the boundary where existing theories break down.",
        "Science progresses through cycles of hypothesis experiment and refinement.",
        "Replication of an experiment by independent researchers confirms a discovery.",
        "A paradigm shift occurs when a new theory replaces an established framework.",
    ],

    # --- Domain: art (beauty, creation, expression) ---
    "art": [
        "Beauty in art arises from the harmony of form color and composition.",
        "Creation transforms raw material and emotion into a work of art.",
        "Expression through art communicates what words alone cannot convey.",
        "The artist pursues beauty through deliberate and intuitive creation.",
        "Art challenges perception and invites the viewer into new expression.",
        "Creation in art requires both technical skill and emotional depth.",
        "Expression is the soul of art and beauty is its visible form.",
        "Throughout history art has served as a vehicle for cultural expression.",
        "The creation of art demands patience vision and a sensitivity to beauty.",
        "Art endures because expression and beauty resonate across generations.",
    ],
}

CORPUS_ID: dict[str, list[str]] = {
    # --- Domain: kerajaan (raja, ratu, kerajaan) ---
    "kerajaan": [
        "Raja memerintah kerajaan dengan kebijaksanaan dan kewibawaan.",
        "Ratu memberikan nasihat kepada raja dalam urusan negara dan diplomasi.",
        "Sebuah kerajaan makmur jika penguasa memerintah dengan keadilan.",
        "Takhta diwariskan dari raja kepada putra mahkota berdasarkan tradisi.",
        "Pengadilan kerajaan mengumpulkan para bangsawan yang mengabdi pada raja dan ratu.",
        "Mahkota melambangkan kekuasaan dan tanggung jawab seorang penguasa.",
        "Raja memerintahkan pasukan untuk mempertahankan kerajaan dari penjajah.",
        "Ratu mengawasi pendidikan para putri dan putra kerajaan.",
        "Titah kerajaan membawa kekuatan hukum di seluruh kerajaan.",
        "Kerajaan merayakan ketika raja kembali menang dari medan perang.",
    ],

    # --- Domain: filsafat (keberadaan, makna, kebenaran) ---
    "filsafat": [
        "Kebenaran adalah tujuan akhir dari penyelidikan dan perenungan filsafat.",
        "Keberadaan mendahului hakikat menurut filsafat eksistensialisme.",
        "Makna kehidupan telah diperdebatkan oleh para pemikir selama ribuan tahun.",
        "Akal dan logika membentuk landasan argumen filsafat.",
        "Kebijaksanaan muncul dari mempertanyakan hakikat kenyataan dan pengetahuan.",
        "Etika mengkaji apa yang membentuk kehidupan yang baik dan bermoral.",
        "Kesadaran tetap menjadi salah satu misteri terdalam dalam filsafat.",
        "Pencarian kebenaran memerlukan keraguan dan pemeriksaan yang ketat.",
        "Kehendak bebas dan determinisme menimbulkan ketegangan filsafat yang mendasar.",
        "Seorang filsuf mencari pemahaman melampaui sekadar pendapat dan keyakinan.",
    ],

    # --- Domain: kedokteran (penyakit, pengobatan, penyembuhan) ---
    "kedokteran": [
        "Dokter mendiagnosis penyakit melalui pemeriksaan dan pengujian yang cermat.",
        "Pengobatan harus mengatasi akar penyebab bukan sekadar meredakan gejala.",
        "Penyembuhan memerlukan intervensi medis dan pemulihan alami tubuh.",
        "Pencegahan penyakit lebih efektif daripada pengobatan setelah serangan.",
        "Pasien menjalani resep yang diberikan untuk sembuh dari sakit.",
        "Pembedahan kadang diperlukan jika obat saja tidak dapat menyembuhkan luka.",
        "Antibiotik melawan infeksi bakteri namun tidak dapat mengobati penyakit virus.",
        "Rumah sakit menyediakan perawatan khusus untuk kondisi yang parah dan kritis.",
        "Penelitian pengobatan baru memajukan batas kemajuan ilmu kedokteran modern.",
        "Diagnosis memastikan sifat penyakit dan menuntun jalannya terapi.",
    ],

    # --- Domain: alam (sungai, gunung, hutan) ---
    "alam": [
        "Sungai mengukir jalan menembus gunung selama ribuan tahun.",
        "Hutan kuno menaungi ribuan spesies di bawah kanopinya yang lebat.",
        "Gunung berdiri sebagai saksi bisu atas berlalunya zaman.",
        "Hutan menyaring udara dan mengatur aliran sungai.",
        "Lembah terletak di antara dua pegunungan yang dialiri sungai berkelok.",
        "Alam menyeimbangkan pertumbuhan dan kematian dalam siklus pembaruan yang tiada henti.",
        "Sungai memelihara hutan dan hutan melindungi sungai.",
        "Satwa liar bergantung pada hutan untuk tempat berlindung dan sungai untuk air.",
        "Ekosistem gunung mendukung kehidupan dari kaki hingga puncak bersalju.",
        "Musim mengubah hutan dan mengubah irama aliran sungai.",
    ],

    # --- Domain: peperangan (strategi, pertempuran, pertahanan) ---
    "peperangan": [
        "Strategi menentukan hasil pertempuran sebelum tembakan pertama dilepaskan.",
        "Pertahanan memerlukan posisi yang diperkuat dan prajurit yang terlatih disiplin.",
        "Jenderal merancang strategi untuk mengakali musuh dalam pertempuran.",
        "Pengepungan menguji ketahanan baik penyerang maupun pertahanan.",
        "Kemenangan dalam pertempuran bergantung pada persiapan dan strategi yang unggul.",
        "Pasukan memperkuat pertahanan di sepanjang perbatasan yang rentan.",
        "Intelijen dan tipu daya adalah unsur penting dalam strategi peperangan.",
        "Pertempuran dapat berbalik jika salah satu pihak memperoleh keunggulan strategis.",
        "Pertahanan memegang garis sementara cadangan mempersiapkan serangan balik.",
        "Sepanjang sejarah peperangan telah mendorong inovasi strategi dan teknologi.",
    ],

    # --- Domain: perdagangan (niaga, pasar, pertukaran) ---
    "perdagangan": [
        "Niaga menghubungkan daerah yang jauh melalui pertukaran barang dan jasa.",
        "Pasar menetapkan harga melalui keseimbangan penawaran dan permintaan.",
        "Pedagang memfasilitasi perdagangan dengan menjembatani produsen dan konsumen.",
        "Pertukaran nilai adalah transaksi mendasar di setiap pasar.",
        "Perdagangan berkembang jika jalur niaga aman dan regulasi adil.",
        "Pasar menghargai efisiensi dan menghukum pemborosan dalam niaga yang kompetitif.",
        "Perdagangan internasional memperluas pasar melampaui batas domestik.",
        "Pertukaran yang adil memerlukan transparansi dan kepercayaan antara mitra niaga.",
        "Perdagangan mendorong kemakmuran dengan memungkinkan spesialisasi dan inovasi.",
        "Pasar menyesuaikan harga untuk mencerminkan perubahan penawaran dan permintaan.",
    ],

    # --- Domain: hukum (keadilan, regulasi, kontrak) ---
    "hukum": [
        "Keadilan adalah landasan tempat supremasi hukum berdiri.",
        "Kontrak mengikat para pihak pada kewajiban dan hak yang disepakati.",
        "Regulasi memastikan bahwa perdagangan berjalan dalam batas yang adil dan sah.",
        "Pengadilan menafsirkan hukum dan memberikan keadilan kepada para pihak.",
        "Kontrak hukum harus bersifat sukarela dan didukung oleh pertimbangan.",
        "Regulasi melindungi masyarakat dari kerugian dan praktik bisnis yang tidak adil.",
        "Hakim menerapkan hukum secara tidak memihak untuk mencapai keadilan.",
        "Pelanggaran kontrak memberikan hak kepada pihak yang dirugikan untuk ganti rugi.",
        "Hukum berkembang melalui undang-undang dan interpretasi yudisial seiring waktu.",
        "Keadilan menghendaki hukum memperlakukan semua orang dengan martabat yang setara.",
    ],

    # --- Domain: sains (eksperimen, teori, penemuan) ---
    "sains": [
        "Eksperimen menguji hipotesis dalam kondisi terkendali dan dapat diulang.",
        "Teori menjelaskan pengamatan dan memprediksi hasil eksperimen yang akan datang.",
        "Penemuan memajukan pengetahuan ilmiah dengan mengungkap fenomena yang belum diketahui.",
        "Metode ilmiah mengharuskan setiap teori dapat difalsifikasi oleh eksperimen.",
        "Eksperimen yang dirancang dengan baik meminimalkan bias dan mengisolasi variabel.",
        "Teori yang kokoh bertahan terhadap pengujian eksperimental dan tinjauan sejawat.",
        "Penemuan sering muncul di batas tempat teori yang ada runtuh.",
        "Sains maju melalui siklus hipotesis eksperimen dan penyempurnaan.",
        "Replikasi eksperimen oleh peneliti independen mengkonfirmasi suatu penemuan.",
        "Pergeseran paradigma terjadi jika teori baru menggantikan kerangka yang mapan.",
    ],

    # --- Domain: seni (keindahan, penciptaan, ekspresi) ---
    "seni": [
        "Keindahan dalam seni lahir dari harmoni bentuk warna dan komposisi.",
        "Penciptaan mengubah bahan mentah dan emosi menjadi karya seni.",
        "Ekspresi melalui seni menyampaikan apa yang tidak bisa dikatakan kata-kata saja.",
        "Seniman mengejar keindahan melalui penciptaan yang disengaja dan intuitif.",
        "Seni menantang persepsi dan mengajak pengamat ke dalam ekspresi baru.",
        "Penciptaan dalam seni menuntut keterampilan teknis dan kedalaman emosional.",
        "Ekspresi adalah jiwa seni dan keindahan adalah wujud kasarnya.",
        "Sepanjang sejarah seni telah menjadi wadah ekspresi budaya.",
        "Penciptaan seni menuntut kesabaran visi dan kepekaan terhadap keindahan.",
        "Seni bertahan karena ekspresi dan keindahan bergema lintas generasi.",
    ],
}

# Domain alignment: maps English domain name → Indonesian domain name
_DOMAIN_ALIGNMENT: dict[str, str] = {
    "royalty": "kerajaan",
    "philosophy": "filsafat",
    "medicine": "kedokteran",
    "nature": "alam",
    "warfare": "peperangan",
    "commerce": "perdagangan",
    "law": "hukum",
    "science": "sains",
    "art": "seni",
}


def get_corpus(lang: str = "en") -> dict[str, list[str]]:
    """Return the corpus for the given language.

    Args:
        lang: Language code — "en" for English, "id" for Indonesian.

    Returns:
        Dictionary mapping domain names to lists of sentences.
    """
    if lang == "id":
        return dict(CORPUS_ID)
    return dict(CORPUS_EN)


def get_aligned_sentences() -> list[tuple[str, str, str]]:
    """Return aligned English-Indonesian sentence pairs.

    Each tuple is (domain, english_sentence, indonesian_sentence) where
    domain is the English domain name. The sentences are aligned by index
    within each domain, creating perfect translation pairs for testing
    convergence detection across languages.

    Returns:
        List of (domain, english, indonesian) tuples.
    """
    result: list[tuple[str, str, str]] = []
    for en_domain, id_domain in _DOMAIN_ALIGNMENT.items():
        en_sentences = CORPUS_EN.get(en_domain, [])
        id_sentences = CORPUS_ID.get(id_domain, [])
        # Align by index — both should have the same number of sentences
        for i in range(min(len(en_sentences), len(id_sentences))):
            result.append((en_domain, en_sentences[i], id_sentences[i]))
    return result
