#include "serdes.h"
#include <cassert>
#include <iostream>

int main() {
	
    int i = 0xff;
	std::string s = "testString";
	std::string t1 = "i", t2 = "s";
    int ret;

	nixlSerDes sd;

	ret = sd.addBuf(t1, &i, sizeof(i));
    assert(ret == 0);

	ret = sd.addStr(t2, s);
    assert(ret == 0);

	std::string sdbuf = sd.exportStr();
    assert(sdbuf.size() > 0);

    std::cout << "exported string: " << sdbuf << "\n";

	// "nixlSDBegin|i   00000004000000ff|s   0000000AtestString|nixlSDEnd
	// |token      |tag|size.  |value.  |tag|size   |          |token


	nixlSerDes sd2;
	ret = sd2.importStr(sdbuf);
    assert(ret == 0);

	size_t osize = sd2.getBufLen(t1);
    assert(osize > 0);

	void *ptr = malloc(osize);
	ret = sd2.getBuf(t1, ptr, osize);
    assert(ret == 0);

	std::string s2 =  sd2.getStr(t2);
    assert(s2.size() > 0);

    assert(*((int*) ptr) == 0xff);
    
	assert(s2.compare("testString") == 0);

	return 0;
}
